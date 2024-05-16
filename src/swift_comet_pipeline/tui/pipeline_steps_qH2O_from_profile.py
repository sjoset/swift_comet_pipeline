import copy

# from functools import partial
from typing import Optional
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval

# from scipy.interpolate import interp1d
# from scipy.optimize import curve_fit
# import pandas as pd

from swift_comet_pipeline.background.background_result import (
    BackgroundResult,
    dict_to_background_result,
)
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.count_rate import CountRate
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, get_filter_parameters
from swift_comet_pipeline.swift.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)
from swift_comet_pipeline.water_production.fluorescence_OH import (
    flux_OH_to_num_OH,
    gfactor_1au,
)
from swift_comet_pipeline.water_production.flux_OH import (
    OH_flux_from_count_rate,
    beta_parameter,
)
from swift_comet_pipeline.water_production.num_OH_to_Q import (
    num_OH_at_r_au_vectorial,
    num_OH_to_Q_vectorial,
)
from swift_comet_pipeline.tui.tui_common import (
    get_selection,
    stacked_epoch_menu,
    wait_for_key,
)
from swift_comet_pipeline.comet.comet_profile import (
    CometRadialProfile,
    count_rate_from_comet_radial_profile,
    extract_comet_radial_median_profile_from_cone,
    radial_profile_to_image,
)


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

        self.fig, self.axes = plt.subplots(2, 3)
        self.uw1_ax = self.axes[0][0]
        self.uvv_ax = self.axes[1][0]
        self.uw1_profile_ax = self.axes[0][1]
        self.uvv_profile_ax = self.axes[1][1]
        self.uw1_sub_ax = self.axes[0][2]
        self.uvv_sub_ax = self.axes[1][2]

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

        self.setup_image_subtraction()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        self.update_plots()

    def setup_image_subtraction(self):
        """create a distance-from-center mesh and plots to hold the subtracted images"""
        img_height, img_width = self.uw1_img.shape
        center_x, center_y = self.image_center.x, self.image_center.y
        xs = np.linspace(0, img_width, num=img_width, endpoint=False)
        ys = np.linspace(0, img_height, num=img_height, endpoint=False)
        x, y = np.meshgrid(xs, ys)
        # the pixel values in the mesh image are the distance from the center, rounded to the nearest integer, so we can use
        # these values as an index to create a radially symmetric image from a 1-dimensional profile
        self.distance_from_center_mesh = np.round(
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        ).astype(int)
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
        self.update_subtracted_image_plot()
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

    def update_subtracted_image_plot(self):
        uw1_sub_img = copy.deepcopy(self.uw1_img)
        uvv_sub_img = copy.deepcopy(self.uvv_img)

        uw1_median_profile_img = radial_profile_to_image(
            profile=self.uw1_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
        )
        uvv_median_profile_img = radial_profile_to_image(
            profile=self.uvv_radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
        )

        self.uw1_subtraction_plot.set_data(uw1_sub_img - uw1_median_profile_img)
        self.uvv_subtraction_plot.set_data(uvv_sub_img - uvv_median_profile_img)

    def update_q_from_profiles(self):
        # if self.uw1_radial_profile is None or self.uvv_radial_profile is None:
        #     return
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
            f"{detection_str} Q: {self.q_h2o.value:3.2e} +/- {self.q_h2o.sigma:3.2e}\nTime from perihelion: {self.time_from_perihelion.to(u.day)}\n"  # type: ignore
            + f"Q absolute upper limit: {self.abs_upper_limit_q_h2o.value:3.2e} +/- {self.abs_upper_limit_q_h2o.sigma:3.2e}"
        )
        self.fig.suptitle(title_str)

    def show(self):
        plt.show()


def profile_selection_plot(
    pipeline_files: PipelineFiles, stacking_method: StackingMethod
) -> Optional[RadialProfileSelectionPlot]:
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    parent_epoch = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    )
    if parent_epoch is None:
        return

    epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        parent_epoch=parent_epoch
    )
    if epoch_subpipeline is None:
        return

    epoch_subpipeline.stacked_epoch.read()
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        print("Error reading epoch!")
        return

    epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].read()
    epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].read()

    uw1_img = epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].data.data
    uvv_img = epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].data.data

    if uw1_img is None or uvv_img is None:
        print("Error loading background-subtracted images!")
        wait_for_key()
        return

    epoch_subpipeline.background_analyses[SwiftFilter.uw1, stacking_method].read()
    epoch_subpipeline.background_analyses[SwiftFilter.uvv, stacking_method].read()
    uw1_bg = dict_to_background_result(
        epoch_subpipeline.background_analyses[SwiftFilter.uw1, stacking_method].data
    )
    uvv_bg = dict_to_background_result(
        epoch_subpipeline.background_analyses[SwiftFilter.uvv, stacking_method].data
    )

    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        return

    rpsp = RadialProfileSelectionPlot(
        epoch=stacked_epoch,
        uw1_img=uw1_img,
        uw1_bg=uw1_bg,
        uvv_img=uvv_img,
        uvv_bg=uvv_bg,
    )
    rpsp.show()

    show_subtracted_profile(
        rpsp,
        km_per_pix=np.mean(stacked_epoch.KM_PER_PIX),
        rh=np.mean(stacked_epoch.HELIO),
        base_q_per_s=rpsp.q_h2o.value,
        delta_au=np.mean(stacked_epoch.OBS_DIS),
        helio_v_kms=np.mean(stacked_epoch.HELIO_V),
    )
    return rpsp


def arcseconds_to_au(arcseconds, delta):
    return delta * 2 * np.pi * arcseconds / (3600.0 * 360)


def show_subtracted_profile(
    rpsp: RadialProfileSelectionPlot,
    km_per_pix,
    rh,
    base_q_per_s,
    delta_au,
    helio_v_kms,
) -> None:
    uw1_profile: CometRadialProfile = rpsp.uw1_radial_profile
    uvv_profile: CometRadialProfile = rpsp.uvv_radial_profile
    redness: DustReddeningPercent = rpsp.dust_redness
    beta = beta_parameter(redness)

    uw1_params = get_filter_parameters(SwiftFilter.uw1)
    uvv_params = get_filter_parameters(SwiftFilter.uvv)

    uvv_to_uw1_cf = uvv_params["cf"] / uw1_params["cf"]

    # convert from count rate to flux
    subtracted_pixels = (
        uw1_profile.pixel_values - beta * uvv_to_uw1_cf * uvv_profile.pixel_values
    )
    positive_mask = subtracted_pixels > 0
    physical_rs = uw1_profile.profile_axis_xs * km_per_pix

    pix = subtracted_pixels[positive_mask]
    rs = physical_rs[positive_mask]

    # print(f"{base_q_per_s=}, running with Q={3*base_q_per_s}")
    model_Q = 1e29
    _, coma = num_OH_at_r_au_vectorial(base_q_per_s=model_Q, helio_r_au=rh)
    vectorial_values = (base_q_per_s / model_Q) * coma.vmr.column_density_interpolation(
        rs * 1000
    )

    # convert pixel signal to column density
    pixel_area_cm2 = (km_per_pix / 1.0e5) ** 2
    # TODO: the 1.0 arcseconds should reflect the mode the image was taken with
    pixel_side_length_cm = (
        arcseconds_to_au(arcseconds=1.0, delta=delta_au) * u.AU
    ).to_value(u.cm)
    pixel_area_cm2 = pixel_side_length_cm**2
    # print(f"{pixel_area_cm2=}")

    # surface brightness = count rate per unit area
    surf_brightness = pix / pixel_area_cm2

    alpha = 1.2750906353215913e-12
    flux = surf_brightness * alpha
    delta_in_cm = (delta_au * u.AU).to_value(u.cm)
    lumi = flux * 4 * np.pi * delta_in_cm**2

    gfactor = gfactor_1au(helio_v_kms=helio_v_kms) / rh**2
    cdens = lumi / gfactor

    # print(cdens / vectorial_values)
    # print(f"{delta_au=}\t{delta_in_cm=}\t{gfactor=}\t{rh=}")

    fig, ax = plt.subplots()
    ax.plot(rs, vectorial_values / 10000, label="vect")
    ax.plot(rs, cdens, label="img")
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.show()


# TODO
def show_subtracted_profile_new(
    rpsp: RadialProfileSelectionPlot,
    km_per_pix,
    rh,
    base_q_per_s,
    delta_au,
    helio_v_kms,
) -> None:
    uw1_profile: CometRadialProfile = rpsp.uw1_radial_profile
    uvv_profile: CometRadialProfile = rpsp.uvv_radial_profile
    redness: DustReddeningPercent = rpsp.dust_redness
    beta = beta_parameter(redness)

    uw1_params = get_filter_parameters(SwiftFilter.uw1)
    uvv_params = get_filter_parameters(SwiftFilter.uvv)

    # print(uw1_profile.pixel_values - beta * uvv_profile.pixel_values)
    # subtracted_pixels = (
    #     uw1_params["cf"] * uw1_params["fwhm"] * uw1_profile.pixel_values
    #     - uvv_params["cf"] * uvv_params["fwhm"] * beta * uvv_profile.pixel_values
    # )

    loss_factor_uw1 = 1.0
    loss_factor_uvv = 1.0

    # convert from count rate to flux
    subtracted_pixels = (
        uw1_params["cf"] * loss_factor_uw1 * uw1_profile.pixel_values
        - uvv_params["cf"] * loss_factor_uvv * beta * uvv_profile.pixel_values
    )
    positive_mask = subtracted_pixels > 0
    physical_rs = uw1_profile.profile_axis_xs * km_per_pix

    pix = subtracted_pixels[positive_mask]
    rs = physical_rs[positive_mask]
    # print(f"{pix=}")

    # df = pd.DataFrame({"r": rs, "subtracted_pixels": pix})
    # df.to_csv("radial_profile.csv")
    # print(f"{km_per_pix=}")

    model_Q = 1e29 / u.s
    _, coma = num_OH_at_r_au_vectorial(base_q_per_s=model_Q, helio_r_au=rh)
    vectorial_values = (
        base_q_per_s / model_Q.value
    ) * coma.vmr.column_density_interpolation(rs * 1000)
    # print(f"{base_q_per_s=}")

    # _, coma_two = num_OH_at_r_au_vectorial(base_q_per_s=2 * base_q_per_s, helio_r_au=rh)
    # vectorial_values_two = coma_two.vmr.column_density_interpolation(rs * 1000)

    # convert pixel signal to column density
    # pixel_area_cm2 = (km_per_pix / 1.0e5) ** 2
    # pixel_area_cm2 = ((km_per_pix * u.km).to_value(u.cm)) ** 2
    # pixel_area_cm2 = 4.25e-4
    # TODO: the 1.0 arcseconds should reflect the mode the image was taken with
    pixel_side_length_cm = (
        arcseconds_to_au(arcseconds=1.0, delta=delta_au) * u.AU
    ).to_value(u.cm)
    # pixel_area_cm2 = np.power(
    #     (arcseconds_to_au(arcseconds=1.0, delta=delta_au) * u.AU).to_value(u.cm), 2
    # )
    pixel_area_cm2 = pixel_side_length_cm**2
    # print(f"{pixel_area_cm2=}")

    # surface brightness = count rate per unit area
    # TODO: the e-16 factor comes from the 'cf' field of the filter data dictionary, adjusted down for loss over time
    # surf_brightness = (pix / pixel_area_cm2) * 1.0e-16
    # surf_brightness = (pix / pixel_area_cm2) / (
    #     2.35e-11**2
    # )  # 1 arcsecond = 2.35e-11 steradian
    # surf_brightness = pix / pixel_area_cm2

    # multiply by solid angle to get surface brightness
    # surf_brightness = pix * pixel_area_m2 * 2.35e-11
    # surf_brightness = pix * pixel_area_cm2 * 2.35e-11
    surf_brightness = pix * pixel_area_cm2

    alpha = 1.2750906353215913e-12
    flux = surf_brightness * alpha
    # print(f"{flux=}")
    # delta_in_cm = (delta_au * u.AU).to_value(u.cm)
    delta_in_cm = (delta_au * u.AU).to_value(u.cm)
    # lumi = flux * 4 * np.pi * delta_in_cm**2
    lumi = flux * 4 * np.pi * delta_in_cm**2
    # print(f"{lumi=}")

    gfactor = gfactor_1au(helio_v_kms=helio_v_kms) / rh**2
    cdens = lumi / gfactor

    # print((vectorial_values / u.m**2).to_value(1 / u.cm**2))
    # print(cdens)
    # print(cdens / vectorial_values)
    # print(f"{delta_au=}\t{delta_in_cm=}\t{gfactor=}\t{rh=}")
    # print(f"{delta_au=}\t{delta_in_cm=}\t{gfactor=}\t{rh=}")

    fig, ax = plt.subplots()
    ax.plot(rs, vectorial_values / 10000, label="vect")
    # ax.plot(rs, vectorial_values_two)
    ax.plot(rs, cdens, label="img")
    ax.legend()
    plt.show()


def qH2O_from_profile_step(
    swift_project_config: SwiftProjectConfig,
) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    rpsp = profile_selection_plot(
        pipeline_files=pipeline_files, stacking_method=stacking_method
    )
    if rpsp is None:
        return
