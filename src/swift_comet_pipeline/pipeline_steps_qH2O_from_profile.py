from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval

from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.count_rate import CountRate
from swift_comet_pipeline.epochs import Epoch
from swift_comet_pipeline.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)
from swift_comet_pipeline.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.flux_OH import OH_flux_from_count_rate, beta_parameter
from swift_comet_pipeline.num_OH_to_Q import num_OH_to_Q_vectorial
from swift_comet_pipeline.tui import get_selection, stacked_epoch_menu, wait_for_key
from swift_comet_pipeline.determine_background import BackgroundResult
from swift_comet_pipeline.comet_profile import (
    CometRadialProfile,
    count_rate_from_comet_radial_profile,
    extract_comet_radial_median_profile_from_cone,
    extract_comet_radial_profile,
)
from swift_comet_pipeline.pipeline_files import PipelineFiles, PipelineProductType


# TODO: add pixel selection and cone angle to initialization so we can save/restore state
class RadialProfileSelectionPlot(object):
    def __init__(
        self,
        epoch: Epoch,
        uw1_img: SwiftUVOTImage,
        uw1_bg: BackgroundResult,
        uvv_img: SwiftUVOTImage,
        uvv_bg: BackgroundResult,
    ):
        self.epoch = epoch
        self.helio_r_au = np.mean(epoch.HELIO)
        self.helio_v_kms = np.mean(epoch.HELIO_V)
        self.delta = np.mean(epoch.OBS_DIS)
        self.km_per_pix = np.mean(epoch.KM_PER_PIX)

        # TODO: add calculation of perihelion from orbital data or as a given from the user
        # this is particular to C/2013US10
        self.perihelion = Time("2015-11-15")
        self.time_from_perihelion = Time(np.mean(epoch.MID_TIME)) - self.perihelion

        self.uw1_img = uw1_img
        self.uw1_bg = uw1_bg
        self.uvv_img = uvv_img
        self.uvv_bg = uvv_bg

        self.image_center = get_uvot_image_center(self.uw1_img)

        self.fig, self.axes = plt.subplots(2, 2)
        self.uw1_ax = self.axes[0][0]
        self.uvv_ax = self.axes[1][0]
        self.uw1_profile_ax = self.axes[0][1]
        self.uvv_profile_ax = self.axes[1][1]

        self.uw1_ax.set_aspect("equal")  # type: ignore
        self.uvv_ax.set_aspect("equal")  # type: ignore

        self.uw1_ax.set_title("Select radial profile")  # type: ignore

        # TODO: add slider for redness percent
        self.dust_redness = DustReddeningPercent(0.0)
        self.beta_parameter = beta_parameter(self.dust_redness)

        # extract profiles in a cone around the selection from -angle to +angle from the profile selection vector
        self.profile_extraction_cone_size_radians = np.pi / 16

        # slider to select cone size
        self.cone_size_slider_ax = self.fig.add_axes([0.25, 0.05, 0.5, 0.03])  # type: ignore
        self.profile_extraction_cone_size_slider = Slider(
            ax=self.cone_size_slider_ax,
            label="cone size",
            valmin=0.0,
            valmax=np.pi,
            valinit=self.profile_extraction_cone_size_radians,
        )
        self.profile_extraction_cone_size_slider.on_changed(self.update_cone_size)

        # Image coordinates for extracting the profile: start at comet center, and stop at arbitrary point away from center for initialization
        self.profile_begin: PixelCoord = self.image_center
        self.profile_end: PixelCoord = PixelCoord(
            x=self.image_center.x + 50, y=self.image_center.y + 50
        )

        # holds the extracted profiles
        self.uw1_radial_profile = None
        self.uvv_radial_profile = None

        # holds the plots for 2d extracted profiles
        self.uw1_profile_plot = None
        self.uvv_profile_plot = None

        self.colormap = "magma"
        self.zscale = ZScaleInterval()
        self.uw1_vmin, self.uw1_vmax = self.zscale.get_limits(self.uw1_img)
        self.uvv_vmin, self.uvv_vmax = self.zscale.get_limits(self.uvv_img)
        self.uw1_img_plot = self.uw1_ax.imshow(  # type: ignore
            self.uw1_img,
            vmin=self.uw1_vmin,
            vmax=self.uw1_vmax,
            origin="lower",
            cmap=self.colormap,
        )
        self.uvv_img_plot = self.uvv_ax.imshow(  # type: ignore
            self.uvv_img,
            vmin=self.uvv_vmin,
            vmax=self.uvv_vmax,
            origin="lower",
            cmap=self.colormap,
        )

        self.initialize_profile_extraction_mpl_elements()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        self.update_plots()

    def onclick(self, event):
        # check that the click was in the image, and handle it if so
        if event.inaxes != self.uw1_ax and event.inaxes != self.uvv_ax:
            return
        rounded_x = int(np.round(event.xdata))
        rounded_y = int(np.round(event.ydata))
        self.profile_end = PixelCoord(x=rounded_x, y=rounded_y)
        self.update_plots()

    def update_cone_size(self, _):
        self.profile_extraction_cone_size_radians = (
            self.profile_extraction_cone_size_slider.val
        )
        self.update_plots()

    def update_plots(self):
        self.update_profile_extraction()
        self.update_profile_plot()
        self.update_q_from_profiles()
        self.fig.canvas.draw_idle()  # type: ignore

    def initialize_profile_extraction_mpl_elements(self):
        self.uw1_extraction_line = plt.Line2D(
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )
        self.uvv_extraction_line = plt.Line2D(
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )

        # "left" edge of cone, each graph needs a separate line object
        self.cone_neg_line_uw1 = plt.Line2D(
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.cone_neg_line_uvv = plt.Line2D(
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.uw1_ax.add_line(self.cone_neg_line_uw1)
        self.uvv_ax.add_line(self.cone_neg_line_uvv)

        self.cone_pos_line_uw1 = plt.Line2D(
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.cone_pos_line_uvv = plt.Line2D(
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.uw1_ax.add_line(self.cone_pos_line_uw1)
        self.uvv_ax.add_line(self.cone_pos_line_uvv)

        self.uw1_ax.add_line(self.uw1_extraction_line)  # type: ignore
        self.uvv_ax.add_line(self.uvv_extraction_line)  # type: ignore

    def update_profile_extraction(self):
        # given our clicked coordinate, figure out the radius and angle direction of the profile
        x0, y0 = self.image_center.x, self.image_center.y
        x1, y1 = self.profile_end.x, self.profile_end.y
        r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        r = int(np.round(r))

        # figure out the angles of the cone edges and the middle
        self.extraction_cone_mid_angle = np.arctan2(y1 - y0, x1 - x0)
        self.extraction_cone_min_angle = (
            self.extraction_cone_mid_angle - self.profile_extraction_cone_size_radians
        )
        self.extraction_cone_max_angle = (
            self.extraction_cone_mid_angle + self.profile_extraction_cone_size_radians
        )

        theta = self.extraction_cone_mid_angle
        cone_size = self.profile_extraction_cone_size_radians

        # for drawing lines from comet center to the edges of the extraction cone
        cone_neg_endpoint = PixelCoord(
            x=r * np.cos(theta - cone_size) + x0,
            y=r * np.sin(theta - cone_size) + y0,
        )
        cone_pos_endpoint = PixelCoord(
            x=r * np.cos(theta + cone_size) + x0,
            y=r * np.sin(theta + cone_size) + y0,
        )

        # get the median profiles in the cone
        self.uw1_radial_profile = extract_comet_radial_median_profile_from_cone(
            img=self.uw1_img,
            comet_center=self.image_center,
            r=r,
            theta=theta,
            cone_size=cone_size,
        )
        self.uvv_radial_profile = extract_comet_radial_median_profile_from_cone(
            img=self.uvv_img,
            comet_center=self.image_center,
            r=r,
            theta=theta,
            cone_size=cone_size,
        )

        self.uw1_extraction_line.set_xdata([x1, x0])
        self.uw1_extraction_line.set_ydata([y1, y0])
        self.uvv_extraction_line.set_xdata([x1, x0])
        self.uvv_extraction_line.set_ydata([y1, y0])

        self.cone_neg_line_uw1.set_xdata([cone_neg_endpoint.x, x0])
        self.cone_neg_line_uvv.set_xdata([cone_neg_endpoint.x, x0])
        self.cone_neg_line_uw1.set_ydata([cone_neg_endpoint.y, y0])
        self.cone_neg_line_uvv.set_ydata([cone_neg_endpoint.y, y0])
        self.cone_pos_line_uw1.set_xdata([cone_pos_endpoint.x, x0])
        self.cone_pos_line_uvv.set_xdata([cone_pos_endpoint.x, x0])
        self.cone_pos_line_uw1.set_ydata([cone_pos_endpoint.y, y0])
        self.cone_pos_line_uvv.set_ydata([cone_pos_endpoint.y, y0])

    def update_profile_plot(self):
        if self.uw1_radial_profile is None or self.uvv_radial_profile is None:
            print(
                "Attempted to update the profile plot without selecting a profile from the image first! Skipping profile plot."
            )
            return

        # have we already plotted a profile? clear it now
        if self.uw1_profile_plot is not None:
            self.uw1_profile_ax.clear()
        if self.uvv_profile_plot is not None:
            self.uvv_profile_ax.clear()

        uw1_pix_to_km = self.uw1_radial_profile.profile_axis_xs[1:] * self.km_per_pix
        uvv_pix_to_km = self.uvv_radial_profile.profile_axis_xs[1:] * self.km_per_pix
        self.uw1_profile_plot = self.uw1_profile_ax.plot(
            uw1_pix_to_km,
            # np.log10(uw1_pix_to_km),
            self.uw1_radial_profile.pixel_values[1:],
            # np.log10(self.uw1_radial_profile.pixel_values[1:]),
        )
        # draw horizontal shaded bars for 1, 2, and 3 sigma background levels: overlaying with alpha values will make lower sigmas darker
        for i in range(1, 4):
            self.uw1_profile_ax.axhspan(
                -i * self.uw1_bg.count_rate_per_pixel.sigma,
                i * self.uw1_bg.count_rate_per_pixel.sigma,
                color="blue",
                alpha=0.05,
            )
        self.uvv_profile_plot = self.uvv_profile_ax.plot(
            uvv_pix_to_km,
            # np.log10(uvv_pix_to_km),
            self.uvv_radial_profile.pixel_values[1:],
            # np.log10(self.uvv_radial_profile.pixel_values[1:]),
        )
        for i in range(1, 4):
            self.uvv_profile_ax.axhspan(
                -i * self.uvv_bg.count_rate_per_pixel.sigma,
                i * self.uvv_bg.count_rate_per_pixel.sigma,
                color="blue",
                alpha=0.05,
            )

    def update_q_from_profiles(self):
        if self.uw1_radial_profile is None or self.uvv_radial_profile is None:
            return
        self.uw1_count_rate = count_rate_from_comet_radial_profile(
            comet_profile=self.uw1_radial_profile, bg=self.uw1_bg.count_rate_per_pixel
        )
        self.uvv_count_rate = count_rate_from_comet_radial_profile(
            comet_profile=self.uvv_radial_profile, bg=self.uvv_bg.count_rate_per_pixel
        )

        self.flux_OH = OH_flux_from_count_rate(
            uw1=self.uw1_count_rate,
            uvv=self.uvv_count_rate,
            beta=self.beta_parameter,
        )

        # if we subtract nothing from the uvv filter, this is an absolute upper limit on the OH present
        self.abs_upper_limit_flux_OH = OH_flux_from_count_rate(
            uw1=self.uw1_count_rate,
            uvv=CountRate(value=0.0, sigma=self.uvv_bg.count_rate_per_pixel.sigma),
            beta=self.beta_parameter,
        )
        # print(self.abs_upper_limit_flux_OH)

        self.num_OH = flux_OH_to_num_OH(
            flux_OH=self.flux_OH,
            helio_r_au=self.helio_r_au,
            helio_v_kms=self.helio_v_kms,
            delta_au=self.delta,
        )

        self.abs_upper_limit_num_OH = flux_OH_to_num_OH(
            flux_OH=self.abs_upper_limit_flux_OH,
            helio_r_au=self.helio_r_au,
            helio_v_kms=self.helio_v_kms,
            delta_au=self.delta,
        )

        self.q_h2o = num_OH_to_Q_vectorial(
            helio_r_au=self.helio_r_au, num_OH=self.num_OH
        )

        self.abs_upper_limit_q_h2o = num_OH_to_Q_vectorial(
            helio_r_au=self.helio_r_au, num_OH=self.abs_upper_limit_num_OH
        )

        detection_str = ""
        if self.q_h2o.value < 0.0:
            detection_str = "(No detection)"
        title_str = (
            f"{detection_str} Q: {self.q_h2o.value:3.2e} +/- {self.q_h2o.sigma:3.2e}\nTime from perihelion: {self.time_from_perihelion.to(u.day)}\n"
            + f"Q absolute upper limit: {self.abs_upper_limit_q_h2o.value:3.2e} +/- {self.abs_upper_limit_q_h2o.sigma:3.2e}"
        )
        self.fig.suptitle(title_str)

    def show(self):
        plt.show()


def profile_selection_plot(
    pipeline_files: PipelineFiles, stacking_method: StackingMethod
) -> None:
    # TODO: change this menu to only show background-subtracted images
    epoch_id = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_be=True
    )
    if epoch_id is None:
        return

    epoch = pipeline_files.read_pipeline_product(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id
    )
    if epoch is None:
        print("Error reading epoch!")
        wait_for_key()
        return

    uw1_sum = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )

    uvv_sum = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uw1_sum is None or uvv_sum is None:
        print("Error loading background-subtracted images!")
        wait_for_key()
        return

    uw1_bg = pipeline_files.read_pipeline_product(
        PipelineProductType.background_analysis,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv_bg = pipeline_files.read_pipeline_product(
        PipelineProductType.background_analysis,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        wait_for_key()
        return

    rpsp = RadialProfileSelectionPlot(
        epoch=epoch, uw1_img=uw1_sum, uw1_bg=uw1_bg, uvv_img=uvv_sum, uvv_bg=uvv_bg
    )
    rpsp.show()

    # TODO: save any results from rpsp here


def qH2O_from_profile_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    profile_selection_plot(
        pipeline_files=pipeline_files, stacking_method=stacking_method
    )
