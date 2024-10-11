import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

# from astropy.stats import sigma_clipped_stats
# from photutils.detection import DAOStarFinder

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from mpl_toolkits import axes_grid1
from photutils.aperture import CircularAperture

from swift_comet_pipeline.comet.comet_center import (
    get_horizons_comet_center,
    get_user_specified_comet_center,
)
from swift_comet_pipeline.comet.comet_center_finding import (
    CometCenterFindingMethod,
    find_comet_center,
)
from swift_comet_pipeline.observationlog.epoch import Epoch, EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import PixelCoord


class EpochImageSlider(Slider):
    def __init__(self, ax, num_images, valfmt="%1d", **kwargs):
        self.facecolor = kwargs.get("facecolor", "w")
        self.activecolor = kwargs.pop("activecolor", "b")
        self.fontsize = kwargs.pop("fontsize", 10)
        self.num_images = num_images
        self.label: str = "Image"
        initial_image_index = 0

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
        self.pageRects = []

        for i in range(num_images):
            facecolor = self.activecolor if i == initial_image_index else self.facecolor
            r = Rectangle(
                xy=(float(i) / num_images, 0),
                width=1.0 / num_images,
                height=1.0,
                transform=ax.transAxes,
                facecolor=facecolor,
            )
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(
                x=float(i) / num_images + 0.5 / num_images,
                y=0.5,
                s=str(i),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.fontsize,
            )
        self.valtext.set_visible(False)

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

    def next_image(self, event):
        image_index = int(self.val) + 1
        if (image_index < self.valmin) or (image_index >= self.num_images):
            return
        self.set_val(image_index)
        self._colorize(image_index)

    def previous_image(self, event):
        image_index = int(self.val) - 1
        if (image_index < self.valmin) or (image_index >= self.num_images):
            return
        self.set_val(image_index)
        self._colorize(image_index)


class EpochImagePlot(object):
    def __init__(self, swift_data: SwiftData, epoch: Epoch, epoch_title: str):
        self.swift_data = swift_data
        self.epoch = epoch
        self.num_images = len(epoch)
        self.epoch_title = epoch_title

        self.zscale = ZScaleInterval()

        # TODO: add buttons to select which method and have these as defaults
        self.comet_center_finding_method_uw1 = (
            CometCenterFindingMethod.aperture_centroid
        )
        self.comet_center_finding_method_uvv = (
            CometCenterFindingMethod.aperture_centroid
        )

        self.setup_plot_elements()

        self.set_image_index(0)
        self.setup_image_plot()
        self.setup_horizons_marker()
        self.setup_selected_comet_center_marker()
        self.update_plot()
        # self.mark_sources()

    def setup_plot_elements(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.18)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)  # type: ignore
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore

        self.slider_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.04])  # type: ignore
        self.slider = EpochImageSlider(
            self.slider_ax, self.num_images, activecolor="orange"
        )
        self.slider.on_changed(self.slider_update)

        self.veto_ax = plt.axes([0.7, 0.025, 0.1, 0.04])  # type: ignore
        self.approve_ax = plt.axes([0.85, 0.025, 0.1, 0.04])  # type: ignore

        self.veto_button = Button(self.veto_ax, "Veto", color="red", hovercolor="0.975")
        self.veto_button.on_clicked(self.veto_current_image)
        self.approve_button = Button(
            self.approve_ax, "Approve", color="green", hovercolor="0.975"
        )
        self.approve_button.on_clicked(self.approve_current_image)

        self.colorbar_axis = axes_grid1.make_axes_locatable(self.ax).append_axes(
            "right", size="5%", pad="2%"
        )

    def setup_image_plot(self):
        self.img_plot = self.ax.imshow(self.current_image, vmin=self.vmin, vmax=self.vmax, origin="lower", cmap=self.colormap_by_veto())  # type: ignore
        self.colorbar = self.fig.colorbar(self.img_plot, cax=self.colorbar_axis)

    def setup_horizons_marker(self):
        self.comet_x_marker = self.ax.axvline(  # type: ignore
            self.horizon_comet_coords.x, color="b", alpha=0.2
        )
        self.comet_y_marker = self.ax.axhline(  # type: ignore
            self.horizon_comet_coords.y, color="b", alpha=0.2
        )

    def setup_selected_comet_center_marker(self):
        self.selected_comet_patch = plt.Circle(  # type: ignore
            (self.selected_comet_coords.x, self.selected_comet_coords.y),
            edgecolor="white",
            fill=False,
            alpha=1.0,
        )
        self.ax.add_patch(self.selected_comet_patch)  # type: ignore
        self.estimated_comet_radius_pix = self.estimate_comet_radius_pix()
        self.selected_comet_patch.set_radius(self.estimated_comet_radius_pix)

    def set_current_comet_center(self, x, y):
        self.selected_comet_coords = self.search_for_comet_centroid(
            centered_on=PixelCoord(x=x, y=y)
        )
        self.epoch.at[self.current_image_index, "USER_CENTER_X"] = (
            self.selected_comet_coords.x
        )
        self.epoch.at[self.current_image_index, "USER_CENTER_Y"] = (
            self.selected_comet_coords.y
        )

    def search_for_comet_centroid(self, centered_on: PixelCoord):
        comet_aperture = CircularAperture(
            positions=[centered_on.x, centered_on.y],
            r=self.estimated_comet_radius_pix,
        )

        if self.current_epoch_row.FILTER == SwiftFilter.uw1:
            method = self.comet_center_finding_method_uw1
        elif self.current_epoch_row.FILTER == SwiftFilter.uvv:
            method = self.comet_center_finding_method_uvv
        else:
            method = CometCenterFindingMethod.pixel_center

        return find_comet_center(
            img=self.current_image,  # type: ignore
            method=method,
            search_aperture=comet_aperture,
        )

    def estimate_comet_radius_pix(self):
        aperture_physical_size_km = 100000
        return aperture_physical_size_km / self.current_epoch_row.KM_PER_PIX

    def set_image_index(self, image_index):
        """collects all of the information about the image at the given index and sets class variables accordingly"""
        self.current_image_index = image_index
        self.current_epoch_row = self.epoch.iloc[self.current_image_index]

        self.current_image_path = (
            self.swift_data.get_uvot_image_directory(
                obsid=self.current_epoch_row.OBS_ID
            )
            / self.current_epoch_row.FITS_FILENAME
        )
        self.current_image = fits.getdata(self.current_image_path, ext=self.current_epoch_row.EXTENSION)  # type: ignore
        self.vmin, self.vmax = self.zscale.get_limits(self.current_image)
        self.read_horizons_comet_coords()
        self.read_user_selected_comet_coords()

    def read_horizons_comet_coords(self):
        """sets horizon_comet_coords to coordinates of the comet center given by horizons for the current image"""
        epoch_row = self.epoch.iloc[self.current_image_index]
        self.horizon_comet_coords = PixelCoord(x=epoch_row.PX, y=epoch_row.PY)

    def read_user_selected_comet_coords(self):
        """sets selected_comet_coords to coordinates of the comet center specified by user if they exist for the current image"""
        comet_coords = get_user_specified_comet_center(row=self.current_epoch_row)
        if comet_coords is None:
            comet_coords = get_horizons_comet_center(row=self.current_epoch_row)
        self.selected_comet_coords = comet_coords

    def update_image_plot(self):
        self.img_plot.set_data(self.current_image)
        image_height, image_width = self.current_image.shape  # type: ignore
        # extent = [left, right, bottom, top]
        self.img_plot.set_extent(extent=[0, image_width, 0, image_height])
        self.img_plot.set_cmap(self.colormap_by_veto())
        self.img_plot.set_clim(vmin=self.vmin, vmax=self.vmax)

    def update_horizons_comet_marker(self):
        self.comet_x_marker.set_xdata([self.horizon_comet_coords.x])
        self.comet_y_marker.set_ydata([self.horizon_comet_coords.y])

    def update_user_comet_marker(self):
        self.selected_comet_patch.center = (
            self.selected_comet_coords.x,
            self.selected_comet_coords.y,
        )

    def update_plot_title(self):
        self.ax.set_title(  # type: ignore
            self.epoch_title
            + "  ("
            + self.current_image_path.name
            + "  extension "
            + str(self.current_epoch_row.EXTENSION)
            + ")  "
            + f"{self.current_epoch_row.EXPOSURE:4.1f}"
            + " s exposure"
        )

    def update_plot(self):
        # if index < 0 or index >= self.num_images:
        #     return

        self.update_image_plot()
        self.update_horizons_comet_marker()
        self.update_user_comet_marker()
        self.update_plot_title()
        self.fig.canvas.draw_idle()  # type: ignore

    def slider_update(self, new_index):
        if int(new_index) == self.current_image_index:
            return
        # TODO: bounds check the index
        self.set_image_index(int(new_index))
        self.update_plot()
        # self.mark_sources()
        return

    def veto_current_image(self, event):
        self.epoch.loc[self.current_image_index, "manual_veto"] = np.True_
        self.update_plot()

    def approve_current_image(self, event):
        self.epoch.loc[self.current_image_index, "manual_veto"] = np.False_
        self.update_plot()

    def colormap_by_veto(self):
        if self.epoch.loc[self.current_image_index, "manual_veto"] is np.True_:
            cmap = "binary"
        else:
            cmap = "magma"
        return cmap

    def on_key_press(self, event):
        if event.key == "l":
            if self.current_image_index < (self.num_images - 1):
                self.slider.next_image(event)
        elif event.key == "h":
            if self.current_image_index > 0:
                self.slider.previous_image(event)
        elif event.key == "v":
            self.veto_current_image(event)
        elif event.key == "a":
            self.approve_current_image(event)

    def onclick(self, event):
        # check that the click was in the image, and handle it if so
        if event.inaxes != self.ax:
            # print("click not handled")
            return
        rounded_x = int(np.round(event.xdata))
        rounded_y = int(np.round(event.ydata))
        self.set_current_comet_center(x=rounded_x, y=rounded_y)
        self.update_plot()

    def show(self):
        plt.show()

    def mark_sources(self):
        pass
        # if self.aperture_patches is not None:
        #     self.ax.patches.clear()  # type: ignore
        # epoch_row = self.epoch.iloc[self.current_image_index]
        # if epoch_row.FILTER == SwiftFilter.uvv:
        #     self.mark_sources_uvv()
        # else:
        #     self.mark_sources_uw1()

    def mark_sources_uw1(self):
        # mean, median, std = sigma_clipped_stats(self.current_image, sigma=3.0)
        # daofind = DAOStarFinder(fwhm=7.0, threshold=5.0 * std)
        # sources = daofind(self.current_image - median)
        # if sources is None:
        #     self.aperture_patches = None
        #     return
        # positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        # apertures = CircularAperture(positions, r=10.0)
        # self.aperture_patches = apertures.plot(self.ax, color="blue", lw=1.0, alpha=0.8)
        pass

    def mark_sources_uvv(self):
        # mean, median, std = sigma_clipped_stats(self.current_image, sigma=3.0)
        # daofind = DAOStarFinder(fwhm=5.0, threshold=5.0 * std)
        # sources = daofind(self.current_image - median)
        # if sources is None:
        #     self.aperture_patches = None
        #     return
        # positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        # apertures = CircularAperture(positions, r=10.0)
        # self.aperture_patches = apertures.plot(self.ax, color="blue", lw=1.5, alpha=0.5)
        pass


def manual_veto(swift_data: SwiftData, epoch: Epoch, epoch_title: str) -> Epoch:
    # group the uvv and uw1 filters together for image viewing
    epoch = epoch.sort_values("FILTER").reset_index(drop=True)
    EpochImagePlot(swift_data=swift_data, epoch=epoch, epoch_title=epoch_title).show()
    epoch = epoch.sort_values("DATE_OBS").reset_index(drop=True)
    return epoch
