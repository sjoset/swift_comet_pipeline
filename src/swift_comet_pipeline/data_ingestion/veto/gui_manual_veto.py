import numpy as np

from astropy.visualization import ZScaleInterval

from astropy.stats import sigma_clipped_stats
from pandas.core.util.hashing import hash_pandas_object
from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from mpl_toolkits import axes_grid1
from photutils.aperture import CircularAperture


from swift_comet_pipeline.data_ingestion.observation_log.comet_center_tracking import (
    get_comet_center_prefer_user_coords,
    get_horizons_comet_center,
    get_user_specified_comet_center,
)
from swift_comet_pipeline.photometry.comet.comet_center_finding import find_comet_center
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    GlobalKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.swift.swift_data import SwiftData


class EpochImageSlider(Slider):
    def __init__(self, ax, num_images, valfmt="%1d", **kwargs):
        self.facecolor = kwargs.get("facecolor", "w")
        self.activecolor = kwargs.pop("activecolor", "b")
        self.fontsize = kwargs.pop("fontsize", 10)
        self.num_images = num_images
        self.label: str = "Image"
        initial_image_index = 0

        self.ax = ax

        super(EpochImageSlider, self).__init__(
            ax=ax,
            label=self.label,
            valmin=0,
            valmax=num_images,
            valinit=initial_image_index,
            valfmt=valfmt,
            **kwargs,
        )

        self.poly.set_visible(False)
        self.vline.set_visible(False)

        self._construct_slider_artists()

        divider = axes_grid1.make_axes_locatable(ax)
        prev_button_axis = divider.append_axes("right", size="5%", pad=0.05)
        forward_button_axis = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = Button(
            ax=prev_button_axis,
            label="prev",
            color=self.facecolor,
            hovercolor=self.activecolor,
        )
        self.button_forward = Button(
            ax=forward_button_axis,
            label="next",
            color=self.facecolor,
            hovercolor=self.activecolor,
        )
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.previous_image)
        self.button_forward.on_clicked(self.next_image)

    def _construct_slider_artists(self):
        self.pageRects = []
        self.pageTexts = []
        for i in range(self.num_images):
            facecolor = self.activecolor if i == self.val else self.facecolor
            r = Rectangle(
                xy=(float(i) / self.num_images, 0),
                width=1.0 / self.num_images,
                height=1.0,
                transform=self.ax.transAxes,
                facecolor=facecolor,
            )
            self.ax.add_artist(r)
            self.pageRects.append(r)
            pt = self.ax.text(
                x=float(i) / self.num_images + 0.5 / self.num_images,
                y=0.5,
                s=str(i),
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=self.fontsize,
            )
            self.pageTexts.append(pt)
        self.valtext.set_visible(False)
        self.valmax = self.num_images
        self.ax.set_xlim(0, self.num_images)

    def _tear_down_slider_artists(self):
        for r in self.pageRects:
            r.remove()
        self.pageRects = []
        for pt in self.pageTexts:
            pt.remove()
        self.pageTexts = []

    def _update(self, event):
        super(EpochImageSlider, self)._update(event)  # type: ignore
        i = int(self.val)
        if i >= self.num_images:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.num_images):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def _is_index_invalid(self, i) -> bool:
        return (i < self.valmin) or (i >= self.num_images)

    def next_image(self, _):
        image_index = int(self.val) + 1
        if self._is_index_invalid(image_index):
            return
        # this triggers the on_changed hook
        self.set_val(image_index)
        self._colorize(image_index)

    def previous_image(self, _):
        image_index = int(self.val) - 1
        if self._is_index_invalid(image_index):
            return
        # this triggers the on_changed hook
        self.set_val(image_index)
        self._colorize(image_index)

    def set_image_index(self, i):
        if self._is_index_invalid(i):
            return
        self.set_val(i)
        self._colorize(i)

    def set_new_image_count(self, new_image_count):
        self._tear_down_slider_artists()
        self.num_images = new_image_count
        self._construct_slider_artists()


class EpochImagePlot(object):

    def __init__(
        self,
        scp: Products,
        epoch_id_default_view: EpochID | None = None,
        inner_aperture_radius: u.Quantity = 50000 * u.km,  # type: ignore
        outer_aperture_radius: u.Quantity = 100000 * u.km,  # type: ignore
    ):
        self.swift_data = SwiftData(data_path=scp.cfg.swift_data_path)
        self.original_obs_log = scp.load_epoch_log()
        assert self.original_obs_log is not None
        self.obs_log = self.original_obs_log.copy()
        self.inner_aperture_radius = inner_aperture_radius
        self.outer_aperture_radius = outer_aperture_radius
        self.veto_to_cmap = {np.True_: "binary", np.False_: "magma"}

        # for marking sources
        self.aperture_patches = None

        self.zscale = ZScaleInterval()
        self.setup_epoch_data()

        # select an epoch and image index to view so we can construct the matplotlib elements
        if epoch_id_default_view is None:
            epoch_index_default_view = 0
        else:
            epoch_index_default_view = self.epoch_id_to_index[epoch_id_default_view]
        self.set_current_epoch_index(epoch_index_default_view)

        self.setup_plot_elements()
        self.update_plot()
        self.mark_sources()

    def setup_epoch_data(self) -> None:
        # epoch slider will use int indices, so we need to sort our observation log and build
        self.obs_log = self.obs_log.sort_values(["epoch_id", "MID_TIME"])

        # epoch_ids will now be sorted in ascending order
        self.epoch_ids = self.obs_log.epoch_id.unique()
        self.epoch_id_to_index = {eid: i for i, eid in enumerate(self.epoch_ids)}
        self.epoch_index_to_id = {v: k for k, v in self.epoch_id_to_index.items()}

        # dictionary of epoch id --> observation log of that epoch, storted by filter
        epoch_dfs = {}
        for epoch_id, epoch_df in self.obs_log.groupby("epoch_id"):
            edf = epoch_df.sort_values("FILTER").reset_index(drop=True)
            epoch_dfs[epoch_id] = edf
        self.epoch_dfs = epoch_dfs

        self.image_indices = {x: 0 for x in self.epoch_ids}

        return

    def set_current_image_index(self, image_index: int) -> None:

        # validate the index first
        if image_index < 0 or image_index > self.current_image_count - 1:
            return

        self.current_image_index = image_index
        self.image_indices[self.current_epoch_id] = self.current_image_index
        self.current_dataframe_index = self.current_epoch_df.index[
            self.current_image_index
        ]
        self.current_obs_log_row = self.current_epoch_df.iloc[image_index]
        self.current_image_path = self.current_obs_log_row.FULL_FITS_PATH
        self.current_image = self.swift_data.get_observation_image(
            obsid=self.current_obs_log_row.OBS_ID,
            image_mode=self.current_obs_log_row.DATAMODE,
            fits_filename=self.current_obs_log_row.FITS_FILENAME,
            extension_id=self.current_obs_log_row.EXTENSION,
        )
        assert self.current_image is not None

        self.current_data_mode = self.current_obs_log_row.DATAMODE
        # adjust scale for event mode images
        if self.current_data_mode == UvotImageMode.event_mode:
            self.current_image = np.log(self.current_image + 1)
        self.vmin, self.vmax = self.zscale.get_limits(self.current_image)

        # x, y = self.current_obs_log_row.PX, self.current_obs_log_row.PY
        horizons_coords = get_horizons_comet_center(row=self.current_obs_log_row)
        if self.current_data_mode == UvotImageMode.event_mode:
            # event mode images are downsampled by a factor of 2: adjust accordingly
            horizons_coords.x, horizons_coords.y = (
                horizons_coords.x / 2,
                horizons_coords.y / 2,
            )
        self.current_horizons_coords = horizons_coords
        # print(f"{self.current_horizons_coords=}")

        # mark the comet using horizons coordinates, or user-input coordinates
        self.current_comet_coords = get_comet_center_prefer_user_coords(
            row=self.current_obs_log_row
        )
        if self.current_data_mode == UvotImageMode.event_mode:
            # event mode images are downsampled by a factor of 2: adjust accordingly
            # check if the comet has a user selection - if it does, we don't scale - they selected those pixels at our current scale
            user_comet_center = get_user_specified_comet_center(
                row=self.current_obs_log_row
            )
            # print(f"{user_comet_center=}")
            if user_comet_center is None:
                user_comet_pixel_scaling = 2
            else:
                user_comet_pixel_scaling = 1
            self.current_comet_coords = PixelCoord(
                x=self.current_comet_coords.x / user_comet_pixel_scaling,
                y=self.current_comet_coords.y / user_comet_pixel_scaling,
            )
        # print(f"{self.current_comet_coords=}")

        # get colormap to use for this based on veto status
        self.current_cmap = self.veto_to_cmap[self.current_obs_log_row.manual_veto]

        # adjust aperture size to reflect new pixel scale
        self.comet_radius_inner_pix = self._physical_to_pixel(
            self.inner_aperture_radius
        )
        self.comet_radius_outer_pix = self._physical_to_pixel(
            self.outer_aperture_radius
        )

    def set_current_epoch_index(self, epoch_index: int) -> None:
        if epoch_index < 0 or epoch_index >= len(self.epoch_ids):
            return

        self.current_epoch_index = epoch_index
        self.current_epoch_id = self.epoch_index_to_id[epoch_index]
        self.current_epoch_df = self.epoch_dfs[self.current_epoch_id]
        self.current_image_count = len(self.current_epoch_df)

        # remember which image we were viewing from that epoch
        self.set_current_image_index(
            image_index=self.image_indices[self.current_epoch_id]
        )

    def build_new_slider(self):
        self.img_slider.set_new_image_count(self.current_image_count)
        self._sync_image_slider()

    def _sync_image_slider(self):
        self.img_slider.set_image_index(self.current_image_index)

    def _physical_to_pixel(self, dist: u.Quantity) -> float:
        dist_km = dist.to_value(u.km)  # type: ignore
        size_in_pixels = dist_km / self.current_obs_log_row.KM_PER_PIX
        return size_in_pixels

    def setup_plot_elements(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.18)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)  # type: ignore
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore

        # self.img_slider_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.04])  # type: ignore
        self.img_slider_ax = self.fig.add_axes([0.1, 0.025, 0.55, 0.04])  # type: ignore
        self.img_slider = EpochImageSlider(
            self.img_slider_ax, self.current_image_count, activecolor="orange"
        )
        self.img_slider.on_changed(self.image_slider_update)

        self.veto_ax = plt.axes([0.7, 0.025, 0.1, 0.04])  # type: ignore
        self.approve_ax = plt.axes([0.85, 0.025, 0.1, 0.04])  # type: ignore

        self.veto_button = Button(self.veto_ax, "Veto", color="red", hovercolor="0.975")
        self.veto_button.on_clicked(self.veto_current_image)
        self.approve_button = Button(
            self.approve_ax, "Approve", color="green", hovercolor="0.975"
        )
        self.approve_button.on_clicked(self.approve_current_image)

        self.img_plot = self.ax.imshow(
            self.current_image,  # type: ignore
            vmin=self.vmin,
            vmax=self.vmax,
            origin="lower",
            cmap=self.current_cmap,
        )

        self.colorbar_axis = axes_grid1.make_axes_locatable(self.ax).append_axes(
            "right", size="5%", pad="2%"
        )
        self.colorbar = self.fig.colorbar(self.img_plot, cax=self.colorbar_axis)

        self.horizons_comet_x_marker = self.ax.axvline(
            self.current_horizons_coords.x, color="b", alpha=0.3
        )
        self.horizons_comet_y_marker = self.ax.axhline(
            self.current_horizons_coords.y, color="b", alpha=0.3
        )

        self.comet_outer_patch = plt.Circle(  # type: ignore
            (self.current_comet_coords.x, self.current_comet_coords.y),
            edgecolor="white",
            fill=False,
            alpha=1.0,
        )
        self.ax.add_patch(self.comet_outer_patch)  # type: ignore
        self.comet_outer_patch.set_radius(self.comet_radius_outer_pix)

        self.comet_inner_patch = plt.Circle(  # type: ignore
            (self.current_comet_coords.x, self.current_comet_coords.y),
            edgecolor="white",
            fill=False,
            alpha=0.4,
        )
        self.ax.add_patch(self.comet_inner_patch)  # type: ignore
        self.comet_inner_patch.set_radius(self.comet_radius_inner_pix)

    def update_image_plot(self):
        self.img_plot.set_data(self.current_image)
        image_height, image_width = self.current_image.shape  # type: ignore
        # extent = [left, right, bottom, top]
        self.img_plot.set_extent(extent=[0, image_width, 0, image_height])  # type: ignore
        self.img_plot.set_cmap(self.current_cmap)
        self.img_plot.set_clim(vmin=self.vmin, vmax=self.vmax)

    def image_slider_update(self, new_index):
        if int(new_index) == self.current_image_index:
            return
        self.set_current_image_index(int(new_index))
        self.update_plot()
        self.mark_sources()
        return

    def veto_current_image(self, _):
        # current row is a copy and is lost on image change
        self.current_obs_log_row.manual_veto = np.True_
        self.current_epoch_df.at[self.current_dataframe_index, "manual_veto"] = np.True_
        self.current_cmap = self.veto_to_cmap[self.current_obs_log_row.manual_veto]
        self.update_plot()

    def approve_current_image(self, _):
        # current row is a copy and is lost on image change
        self.current_obs_log_row.manual_veto = np.False_
        self.current_epoch_df.at[self.current_image_index, "manual_veto"] = np.False_
        self.current_cmap = self.veto_to_cmap[self.current_obs_log_row.manual_veto]
        self.update_plot()

    def search_for_comet_centroid(self, centered_on: PixelCoord):
        comet_aperture = CircularAperture(
            positions=[centered_on.x, centered_on.y],
            r=self.comet_radius_inner_pix,
        )

        # TODO: remove?
        # if self.current_obs_log_row.FILTER == UvotFilter.uw1:
        #     method = self.comet_center_finding_method_uw1
        # elif self.current_obs_log_row.FILTER == UvotFilter.uvv:
        #     method = self.comet_center_finding_method_uvv
        # else:
        #     method = CometCenterFindingMethod.pixel_center

        return find_comet_center(
            img=self.current_image,  # type: ignore
            method=CometCenterFindingMethod.aperture_peak,
            search_aperture=comet_aperture,
        )

    def update_comet_center(self, x, y):
        selected_comet_coords = self.search_for_comet_centroid(
            centered_on=PixelCoord(x=x, y=y)
        )
        self.current_obs_log_row.at.USER_CENTER_X = selected_comet_coords.x
        self.current_epoch_df.at[self.current_dataframe_index, "USER_CENTER_X"] = (
            selected_comet_coords.x
        )
        self.current_obs_log_row.at.USER_CENTER_Y = selected_comet_coords.y
        self.current_epoch_df.at[self.current_dataframe_index, "USER_CENTER_Y"] = (
            selected_comet_coords.y
        )

        self.current_comet_coords = selected_comet_coords
        return

    def update_comet_aperture_marker(self):
        self.comet_outer_patch.center = (
            self.current_comet_coords.x,
            self.current_comet_coords.y,
        )
        self.comet_inner_patch.center = (
            self.current_comet_coords.x,
            self.current_comet_coords.y,
        )

    def update_horizons_comet_marker(self):
        self.horizons_comet_x_marker.set_xdata([self.current_horizons_coords.x])
        self.horizons_comet_y_marker.set_ydata([self.current_horizons_coords.y])

    def on_key_press(self, event):
        if event.key == "l":
            if self.current_image_index < (self.current_image_count - 1):
                self.img_slider.next_image(event)
        elif event.key == "h":
            if self.current_image_index > 0:
                self.img_slider.previous_image(event)
        elif event.key == "v":
            self.veto_current_image(event)
        elif event.key == "a":
            self.approve_current_image(event)
        elif event.key == "t":
            self.set_current_epoch_index(self.current_epoch_index - 1)
            self.build_new_slider()
            self.update_plot()
        elif event.key == "n":
            self.set_current_epoch_index(self.current_epoch_index + 1)
            self.build_new_slider()
            self.update_plot()

    def onclick(self, event):
        # check that the click was in the image, and handle it if so
        if event.inaxes != self.ax:
            return
        rounded_x = int(np.round(event.xdata))
        rounded_y = int(np.round(event.ydata))
        # print(f"Updating comet center to {rounded_x=}, {rounded_y=}")
        self.update_comet_center(x=rounded_x, y=rounded_y)
        self.update_plot()

    def update_plot_title(self):
        self.ax.set_title(  # type: ignore
            self.current_epoch_id
            + "  ("
            + self.current_image_path.name
            + "  extension "
            + str(self.current_obs_log_row.EXTENSION)
            + ")  "
            + f"{self.current_obs_log_row.EXPOSURE:4.1f}"
            + " s exposure"
            + f"  ({self.current_obs_log_row.KM_PER_PIX:4.1f} km/pix)"
            + f"  delta: {self.current_obs_log_row.OBS_DIS:3.2} AU"
        )

    def update_plot(self):
        self.update_image_plot()
        self.update_horizons_comet_marker()
        self.update_comet_aperture_marker()
        self.update_plot_title()

        self.fig.canvas.draw_idle()  # type: ignore
        return

    def show(self):
        plt.show()

    def mark_sources(self):
        if self.aperture_patches is not None:
            # self.ax.patches.clear()  # type: ignore
            for ap in self.aperture_patches:
                ap.remove()
            self.aperture_patches = None

        # if self.current_obs_log_row.FILTER == UvotFilter.uvv:
        #     self.mark_sources_uvv()
        # too slow
        # elif self.current_obs_log_row.FILTER == UvotFilter.uw1:
        #     self.mark_sources_uw1()

    def mark_sources_uvv(self):
        assert self.current_image is not None
        _, median, std = sigma_clipped_stats(self.current_image, sigma=3.0)
        daofind = DAOStarFinder(fwhm=5.0, threshold=5.0 * std)
        sources = daofind(self.current_image - median)
        if sources is None:
            self.aperture_patches = None
            return
        positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        apertures = CircularAperture(positions, r=10.0)
        self.aperture_patches = apertures.plot(self.ax, color="blue", lw=1.5, alpha=0.5)

    # def mark_sources_uw1(self):
    #     assert self.current_image is not None
    #     mean, median, std = sigma_clipped_stats(self.current_image, sigma=3.0)
    #     daofind = DAOStarFinder(fwhm=30.0, threshold=7.0 * std)
    #     sources = daofind(self.current_image - median)
    #     if sources is None:
    #         self.aperture_patches = None
    #         return
    #     positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    #     apertures = CircularAperture(positions, r=10.0)
    #     self.aperture_patches = apertures.plot(self.ax, color="blue", lw=1.5, alpha=0.5)


def manual_veto(scp: Products) -> None:
    # we have to do every epoch in one call to return a finished log for writing

    print("Starting veto..")
    eip = EpochImagePlot(scp=scp)
    eip.show()

    # TODO: EpochImagePlot only needs SwiftData and observation log for arguments
    # TODO: add toggle for identifying and marking sources

    # reassemble the observation log from the altered epochs during veto
    orig_df = eip.original_obs_log
    assert orig_df is not None
    veto_df = pd.concat(eip.epoch_dfs.values())

    veto_df = veto_df.sort_values(["epoch_id", "MID_TIME"]).reset_index(drop=True)
    orig_df = orig_df.sort_values(["epoch_id", "MID_TIME"]).reset_index(drop=True)
    print(f"Observation log comparison: {veto_df.compare(orig_df)}")

    hash_before = hash_pandas_object(orig_df).sum()
    hash_after = hash_pandas_object(veto_df).sum()
    print(f"Hash before: {hash_before}")
    print(f"Hash after: {hash_after}")

    if hash_before != hash_after:
        print("Change detected! Writing.")
        scp.save_obs_log(df=veto_df)
    else:
        print("No change detected!")
        if not scp.exists(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
            )
        ):
            print("Writing vetoed dataframe..")
            scp.save_obs_log(df=veto_df)
            print("Done writing.")
        else:
            print("Not writing.")

    return None
