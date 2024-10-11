import copy

from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from rich import print as rprint

from swift_comet_pipeline.background.background_result import (
    BackgroundResult,
    yaml_dict_to_background_result,
)
from swift_comet_pipeline.observationlog.stacked_epoch import StackedEpoch
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.count_rate import CountRate
from swift_comet_pipeline.observationlog.epoch import epoch_stacked_image_to_fits
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)
from swift_comet_pipeline.tui.tui_menus import subpipeline_selection_menu
from swift_comet_pipeline.water_production.fluorescence_OH import (
    flux_OH_to_num_OH,
)
from swift_comet_pipeline.water_production.flux_OH import (
    OH_flux_from_count_rate,
    beta_parameter,
)
from swift_comet_pipeline.water_production.num_OH_to_Q import (
    num_OH_to_Q_vectorial,
)
from swift_comet_pipeline.tui.tui_common import (
    get_selection,
    wait_for_key,
)
from swift_comet_pipeline.comet.comet_radial_profile import (
    calculate_distance_from_center_mesh,
    count_rate_from_comet_radial_profile,
    extract_comet_radial_median_profile_from_cone,
    radial_profile_to_dataframe_product,
    radial_profile_to_image,
)


# TODO: move this somewhere else
# TODO: add pixel selection and cone angle to initialization so we can save/restore state
class RadialProfileSelectionPlot(object):
    def __init__(
        self,
        stacked_epoch: StackedEpoch,
        uw1_img: SwiftUVOTImage,
        uw1_bg: BackgroundResult,
        uvv_img: SwiftUVOTImage,
        uvv_bg: BackgroundResult,
        t_perihelion: Time,
    ):
        self.stacked_epoch = stacked_epoch
        self.helio_r_au = np.mean(stacked_epoch.HELIO)
        self.helio_v_kms = np.mean(stacked_epoch.HELIO_V)
        self.delta = np.mean(stacked_epoch.OBS_DIS)
        self.km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)

        self.t_perihelion = t_perihelion
        self.time_from_perihelion = (
            Time(np.mean(stacked_epoch.MID_TIME)) - self.t_perihelion
        )

        self.uw1_img = uw1_img
        self.uw1_bg = uw1_bg
        self.uvv_img = uvv_img
        self.uvv_bg = uvv_bg

        self.image_center = get_uvot_image_center(self.uw1_img)

        self.fig, self.axes = plt.subplots(2, 4)
        self.uw1_ax = self.axes[0][0]
        self.uvv_ax = self.axes[1][0]
        self.uw1_profile_ax = self.axes[0][1]
        self.uvv_profile_ax = self.axes[1][1]
        self.uw1_sub_ax = self.axes[0][2]
        self.uvv_sub_ax = self.axes[1][2]
        self.uw1_div_ax = self.axes[0][3]
        self.uvv_div_ax = self.axes[1][3]

        self.uw1_ax.set_aspect("equal")  # type: ignore
        self.uvv_ax.set_aspect("equal")  # type: ignore

        self.uw1_ax.set_title("Select radial profile")  # type: ignore

        self.setup_redness_slider(initial_redness=0.0)

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

        self.setup_mesh()
        self.setup_image_subtraction()
        self.setup_image_division()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        self.update_plots()

    # TODO: constructor to initialize from a finished analysis
    # @classmethod
    # def from_saved_state(self, ...):

    def setup_mesh(self):
        """
        Create a distance-from-center mesh we can calculate once and store because the images do not change size
        """
        self.distance_from_center_mesh = calculate_distance_from_center_mesh(
            img=self.uw1_img
        )

    def setup_image_subtraction(self):
        """create plots to hold the median-subtracted images"""
        self.uw1_subtraction_plot = self.uw1_sub_ax.imshow(  # type: ignore
            self.uw1_img,
            vmin=self.uw1_vmin,
            vmax=self.uw1_vmax,
            origin="lower",
            cmap=self.colormap,
        )
        self.uvv_subtraction_plot = self.uvv_sub_ax.imshow(  # type: ignore
            self.uvv_img,
            vmin=self.uvv_vmin,
            vmax=self.uvv_vmax,
            origin="lower",
            cmap=self.colormap,
        )

    def setup_image_division(self):
        """create plots to hold the median-divided images"""
        # TODO: magic numbers for the min/max scale
        self.uw1_division_plot = self.uw1_div_ax.imshow(
            self.uw1_img,
            vmin=-0.001,
            vmax=10,
            origin="lower",
            cmap=self.colormap,
        )
        self.uvv_division_plot = self.uvv_div_ax.imshow(
            self.uvv_img,
            vmin=-0.001,
            vmax=10,
            origin="lower",
            cmap=self.colormap,
        )

    def setup_redness_slider(self, initial_redness):
        self.redness_slider_ax = self.fig.add_axes([0.25, 0.02, 0.5, 0.03])  # type: ignore
        self.redness_slider = Slider(
            ax=self.redness_slider_ax,
            label="redness",
            valmin=0.0,
            valmax=100.0,
            valinit=initial_redness,
        )
        self.redness_slider.on_changed(self.update_redness)
        self.dust_redness = initial_redness
        self.beta_parameter = beta_parameter(self.dust_redness)

    def update_redness(self, redness: DustReddeningPercent):
        self.dust_redness = redness
        self.beta_parameter = beta_parameter(self.dust_redness)
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
        self.update_median_subtracted_image_plot()
        self.update_median_divided_image_plot()
        self.fig.canvas.draw_idle()  # type: ignore

    def initialize_profile_extraction_mpl_elements(self):
        self.uw1_extraction_line = plt.Line2D(  # type: ignore
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )
        self.uvv_extraction_line = plt.Line2D(  # type: ignore
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )

        # "left" edge of cone, each graph needs a separate line object
        self.cone_neg_line_uw1 = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.cone_neg_line_uvv = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.uw1_ax.add_line(self.cone_neg_line_uw1)
        self.uvv_ax.add_line(self.cone_neg_line_uvv)

        self.cone_pos_line_uw1 = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.cone_pos_line_uvv = plt.Line2D(  # type: ignore
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

        # update middle line of cone
        self.uw1_extraction_line.set_xdata([x1, x0])
        self.uw1_extraction_line.set_ydata([y1, y0])
        self.uvv_extraction_line.set_xdata([x1, x0])
        self.uvv_extraction_line.set_ydata([y1, y0])

        # update bounding lines of the cone
        self.cone_neg_line_uw1.set_xdata([cone_neg_endpoint.x, x0])
        self.cone_neg_line_uvv.set_xdata([cone_neg_endpoint.x, x0])
        self.cone_neg_line_uw1.set_ydata([cone_neg_endpoint.y, y0])
        self.cone_neg_line_uvv.set_ydata([cone_neg_endpoint.y, y0])
        self.cone_pos_line_uw1.set_xdata([cone_pos_endpoint.x, x0])
        self.cone_pos_line_uvv.set_xdata([cone_pos_endpoint.x, x0])
        self.cone_pos_line_uw1.set_ydata([cone_pos_endpoint.y, y0])
        self.cone_pos_line_uvv.set_ydata([cone_pos_endpoint.y, y0])

    def update_profile_plot(self):
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

    def update_median_subtracted_image_plot(self):
        uw1_sub_img = copy.deepcopy(self.uw1_img)
        uvv_sub_img = copy.deepcopy(self.uvv_img)

        self.uw1_median_profile_img = radial_profile_to_image(
            profile=self.uw1_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=0.0,
        )
        self.uvv_median_profile_img = radial_profile_to_image(
            profile=self.uvv_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=0.0,
        )

        self.uw1_subtracted_median_image = uw1_sub_img - self.uw1_median_profile_img
        self.uvv_subtracted_median_image = uvv_sub_img - self.uvv_median_profile_img

        uw1_div_img = radial_profile_to_image(
            profile=self.uw1_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=1.0,
        )
        uvv_div_img = radial_profile_to_image(
            profile=self.uvv_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=1.0,
        )
        self.uw1_divided_median_image = uw1_sub_img / uw1_div_img
        self.uvv_divided_median_image = uvv_sub_img / uvv_div_img

        self.uw1_subtraction_plot.set_data(self.uw1_subtracted_median_image)
        self.uvv_subtraction_plot.set_data(self.uvv_subtracted_median_image)

    def update_median_divided_image_plot(self):
        uw1_sub_img = copy.deepcopy(self.uw1_img)
        uvv_sub_img = copy.deepcopy(self.uvv_img)

        # fill non-profile pixels with 1 because we are dividing by this image
        uw1_div_img = radial_profile_to_image(
            profile=self.uw1_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=1.0,
        )
        uvv_div_img = radial_profile_to_image(
            profile=self.uvv_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=1.0,
        )
        self.uw1_divided_median_image = uw1_sub_img / uw1_div_img
        self.uvv_divided_median_image = uvv_sub_img / uvv_div_img

        # self.uw1_division_plot.norm.vmin, self.uw1_division_plot.norm.vmax = (
        #     self.zscale.get_limits(self.uw1_divided_median_image)
        # )
        # self.uvv_division_plot.norm.vmin, self.uvv_division_plot.norm.vmax = (
        #     self.zscale.get_limits(self.uvv_divided_median_image)
        # )

        self.uw1_division_plot.set_data(self.uw1_divided_median_image)
        self.uvv_division_plot.set_data(self.uvv_divided_median_image)

    def update_q_from_profiles(self):
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
            helio_r_au=self.helio_r_au,
            num_OH=self.abs_upper_limit_num_OH,
        )

        detection_str = ""
        if self.q_h2o.value < 0.0:
            detection_str = "(No detection)"
        title_str = (
            f"{detection_str} Q: {self.q_h2o.value:3.2e} +/- {self.q_h2o.sigma:3.2e}\nTime from perihelion: {self.time_from_perihelion.to(u.day)}\n"  # type: ignore
            + f"Q absolute upper limit: {self.abs_upper_limit_q_h2o.value:3.2e} +/- {self.abs_upper_limit_q_h2o.sigma:3.2e}"
        )
        self.fig.suptitle(title_str)

    def show(self):
        plt.show()


def profile_selection_plot(
    scp: SwiftCometPipeline, stacking_method: StackingMethod
) -> RadialProfileSelectionPlot | None:

    selected_epoch_id = subpipeline_selection_menu(
        scp=scp, status_marker=SwiftCometPipelineStepEnum.vectorial_analysis
    )
    if selected_epoch_id is None:
        return None

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=selected_epoch_id
    )
    if stacked_epoch is None:
        return None

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return None
    t_perihelion = t_perihelion_list[0].t_perihelion

    uw1_img = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_img is not None
    uw1_img = uw1_img.data
    uvv_img = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_img is not None
    uvv_img = uvv_img.data

    if uw1_img is None or uvv_img is None:
        print("Error loading background-subtracted images!")
        wait_for_key()
        return

    uw1_bg = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_bg is not None
    uw1_bg = yaml_dict_to_background_result(raw_yaml=uw1_bg)
    uvv_bg = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_bg is not None
    uvv_bg = yaml_dict_to_background_result(raw_yaml=uvv_bg)

    rpsp = RadialProfileSelectionPlot(
        stacked_epoch=stacked_epoch,
        uw1_img=uw1_img,
        uw1_bg=uw1_bg,
        uvv_img=uvv_img,
        uvv_bg=uvv_bg,
        t_perihelion=t_perihelion,
    )
    rpsp.show()

    rprint("[green]Writing extracted profiles for uw1 and uvv filters...[/green]")

    uw1_profile = scp.get_product(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_profile is not None
    uw1_profile.data = radial_profile_to_dataframe_product(
        profile=rpsp.uw1_radial_profile, km_per_pix=rpsp.km_per_pix
    )
    uw1_profile.write()
    uvv_profile = scp.get_product(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_profile is not None
    uvv_profile.data = radial_profile_to_dataframe_product(
        profile=rpsp.uvv_radial_profile, km_per_pix=rpsp.km_per_pix
    )
    uvv_profile.write()

    prof_img_uw1 = scp.get_product(
        pf=PipelineFilesEnum.extracted_profile_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert prof_img_uw1 is not None
    prof_img_uw1.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uw1_median_profile_img
    )
    prof_img_uw1.write()
    prof_img_uvv = scp.get_product(
        pf=PipelineFilesEnum.extracted_profile_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert prof_img_uvv is not None
    prof_img_uvv.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uvv_median_profile_img
    )
    prof_img_uvv.write()

    med_sub_uw1 = scp.get_product(
        pf=PipelineFilesEnum.median_subtracted_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert med_sub_uw1 is not None
    med_sub_uw1.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uw1_subtracted_median_image
    )
    med_sub_uw1.write()
    med_sub_uvv = scp.get_product(
        pf=PipelineFilesEnum.median_subtracted_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert med_sub_uvv is not None
    med_sub_uvv.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uvv_subtracted_median_image
    )
    med_sub_uvv.write()

    med_div_uw1 = scp.get_product(
        pf=PipelineFilesEnum.median_divided_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert med_div_uw1 is not None
    med_div_uw1.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uw1_divided_median_image
    )
    med_div_uw1.write()
    med_div_uvv = scp.get_product(
        pf=PipelineFilesEnum.median_divided_image,
        epoch_id=selected_epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert med_div_uvv is not None
    med_div_uvv.data = epoch_stacked_image_to_fits(
        epoch=stacked_epoch, img=rpsp.uvv_divided_median_image
    )
    med_div_uvv.write()

    # TODO: write a dataclass to hold the q_from_profile_extraction and associated to_dict and from_dict for saving/loading if needed
    # TODO: write the code to fill in this dataclass and save the product

    return rpsp


def vectorial_analysis_step(
    swift_project_config: SwiftProjectConfig,
) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    rpsp = profile_selection_plot(scp=scp, stacking_method=stacking_method)
    if rpsp is None:
        return
