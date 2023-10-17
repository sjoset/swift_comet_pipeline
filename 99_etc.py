#!/usr/bin/env python3

import os
import pathlib
import warnings
import sys
import logging as log
import numpy as np

import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from argparse import ArgumentParser
from comet_profile import (
    count_rate_from_comet_radial_profile,
    extract_comet_radial_profile,
)

from configs import read_swift_project_config
from count_rate import CountRatePerPixel
from determine_background import BackgroundResult, yaml_dict_to_background_analysis
from fluorescence_OH import flux_OH_to_num_OH
from flux_OH import OH_flux_from_count_rate, beta_parameter
from num_OH_to_Q import NumQH2O, num_OH_to_Q_vectorial
from pipeline_files import PipelineFiles
from reddening_correction import DustReddeningPercent
from stacking import StackingMethod
from swift_data import SwiftData

from epochs import Epoch, read_epoch
from swift_filter import SwiftFilter
from tui import epoch_menu, stacked_epoch_menu

from astropy.coordinates import get_sun
from astropy.time import Time
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.wcs import FITSFixedWarning

from uvot_image import PixelCoord, SwiftUVOTImage, get_uvot_image_center


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config",
        nargs="?",
        help="Filename of project config",
        default="config.yaml",
    )

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def show_fits_sun_direction(img, sun_x, sun_y, comet_x, comet_y):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img)

    im1 = ax1.imshow(img, vmin=vmin, vmax=vmax)  # type: ignore
    # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    # img_center = get_uvot_image_center(img)

    # ax1.plot([sun_x, image_center_col, -sun_x], [sun_y, image_center_row, -sun_y])
    ax1.plot([sun_x, comet_x], [sun_y, comet_y])  # type: ignore
    ax1.set_xlim(0, img.shape[1])  # type: ignore
    ax1.set_ylim(0, img.shape[0])  # type: ignore
    # ax1.axvline(image_center_col, color="b", alpha=0.2)
    # ax1.axhline(image_center_row, color="b", alpha=0.2)

    # hdu = fits.PrimaryHDU(dust_subtracted)
    # hdu.writeto("subtracted.fits", overwrite=True)
    plt.show()


def find_sun_direction(swift_data: SwiftData, pipeline_files: PipelineFiles) -> None:
    """Selects an epoch and attempts to find the direction of the sun on the raw images"""
    epoch_prod = epoch_menu(pipeline_files=pipeline_files)
    if epoch_prod is None:
        return
    epoch_path = epoch_prod.product_path
    epoch_pre_veto = read_epoch(epoch_path)

    filter_mask = epoch_pre_veto["FILTER"] == SwiftFilter.uvv
    epoch_pre_veto = epoch_pre_veto[filter_mask]

    for _, row in epoch_pre_veto.iterrows():
        img = swift_data.get_uvot_image(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        wcs = swift_data.get_uvot_image_wcs(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        print(wcs)

        print(f"t = {Time(row.MID_TIME)}")
        sun = get_sun(Time(row.MID_TIME))
        print(f"{sun=}")

        # print(wcs.wcs_world2pix(sun))
        sun_x, sun_y = skycoord_to_pixel(sun, wcs)
        print(sun_x, sun_y, np.degrees(np.arctan2(sun_y, sun_x)))

        show_fits_sun_direction(img, sun_x, sun_y, row.PX, row.PY)


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
        # plt.subplots_adjust(left=0.20, bottom=0.20)

        self.uw1_ax.set_aspect("equal")  # type: ignore
        self.uvv_ax.set_aspect("equal")  # type: ignore

        self.uw1_ax.set_title("Select radial profile")  # type: ignore
        # self.uvv_ax.set_title("Select radial profile")  # type: ignore

        # self.count_rate_annotation = self.ax.annotate(  # type: ignore
        #     self.count_rate_string(),
        #     xy=(0.03, 0.95),
        #     xytext=(0.03, 0.95),
        #     textcoords="axes fraction",
        #     size=10,
        #     color="orange",
        # )

        # self.radius_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="orange")
        # self.radius_slider = Slider(
        #     self.radius_slider_ax, "r", valmin=1, valmax=100, valstep=1, valinit=50
        # )
        # self.radius_slider.on_changed(self.onslider)
        self.dust_redness = DustReddeningPercent(0.0)
        self.beta_parameter = beta_parameter(self.dust_redness)

        # self.aperture = plt.Circle(
        #     (self.image_center_x, self.image_center_y),
        #     edgecolor="white",
        #     fill=False,
        #     alpha=0.3,
        # )
        # self.ax.add_patch(self.aperture)  # type: ignore
        #
        # self.aperture.set_radius(self.radius_slider.val)

        # Image coordinates for extracting the profile: start at comet center, and stop at arbitrary point away from center for initialization
        self.profile_begin: PixelCoord = self.image_center
        self.profile_end: PixelCoord = PixelCoord(
            x=self.image_center.x + 50, y=self.image_center.y + 50
        )

        # holds the line objects that mark the profile we are looking at
        self.uw1_extraction_line = None
        self.uvv_extraction_line = None

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

    def update_plots(self):
        self.update_profile_extraction()
        self.update_profile_plot()
        self.update_q_from_profiles()
        self.fig.canvas.draw_idle()  # type: ignore

    def update_profile_extraction(self):
        x0, y0 = self.image_center.x, self.image_center.y
        x1, y1 = self.profile_end.x, self.profile_end.y
        r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        r = int(np.round(r))
        theta = np.arctan2(y1 - y0, x1 - x0)

        self.uw1_radial_profile = extract_comet_radial_profile(
            img=self.uw1_img, comet_center=self.image_center, r=r, theta=theta
        )
        self.uvv_radial_profile = extract_comet_radial_profile(
            img=self.uvv_img, comet_center=self.image_center, r=r, theta=theta
        )

        # theta = theta * 180 / np.pi
        # print(f"{r=}, {theta=}")

        if self.uw1_extraction_line is None or self.uvv_extraction_line is None:
            self.uw1_extraction_line = plt.Line2D(
                xdata=[x1, x0], ydata=[y1, y0], lw=1, color="white", alpha=0.3
            )
            self.uvv_extraction_line = plt.Line2D(
                xdata=[x1, x0], ydata=[y1, y0], lw=1, color="white", alpha=0.3
            )
            self.uw1_ax.add_line(self.uw1_extraction_line)  # type: ignore
            self.uvv_ax.add_line(self.uvv_extraction_line)  # type: ignore
        else:
            self.uw1_extraction_line.set_xdata([x1, x0])
            self.uw1_extraction_line.set_ydata([y1, y0])
            self.uvv_extraction_line.set_xdata([x1, x0])
            self.uvv_extraction_line.set_ydata([y1, y0])

        # self.bg_count_rate = bgresult.count_rate_per_pixel.value
        # self.count_rate_annotation.set_text(self.count_rate_string())
        # self.img_plot.set_data(self.original_img - self.bg_count_rate)

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

        self.uw1_profile_plot = self.uw1_profile_ax.plot(
            np.log(self.uw1_radial_profile.profile_axis_xs[1:]),
            self.uw1_radial_profile.pixel_values[1:],
        )
        self.uvv_profile_plot = self.uvv_profile_ax.plot(
            np.log(self.uvv_radial_profile.profile_axis_xs[1:]),
            self.uvv_radial_profile.pixel_values[1:],
        )

    # def onslider(self, new_value):
    #     self.aperture.set_radius(new_value)
    #     self.recalc_background()
    #     self.fig.canvas.draw_idle()  # type: ignore

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

        self.num_OH = flux_OH_to_num_OH(
            flux_OH=self.flux_OH,
            helio_r_au=self.helio_r_au,
            helio_v_kms=self.helio_v_kms,
            delta_au=self.delta,
        )

        self.q_h2o = num_OH_to_Q_vectorial(
            helio_r_au=self.helio_r_au, num_OH=self.num_OH
        )

        if self.q_h2o.value < 0.0:
            self.fig.suptitle(
                f"Q: No detection {self.q_h2o.value:3.2e} +/- {self.q_h2o.sigma:3.2e}"
            )
        else:
            self.fig.suptitle(f"Q: {self.q_h2o.value:3.2e} +/- {self.q_h2o.sigma:3.2e}")

    def show(self):
        plt.show()


def profile_test_plot(pipeline_files: PipelineFiles) -> None:
    stacking_method = StackingMethod.median

    epoch_path = stacked_epoch_menu(pipeline_files=pipeline_files)
    if epoch_path is None:
        return
    if pipeline_files.analysis_bg_subtracted_images is None:
        return

    if pipeline_files.stacked_epoch_products is None:
        return
    epoch_prod = pipeline_files.stacked_epoch_products[epoch_path]
    epoch_prod.load_product()
    epoch = epoch_prod.data_product

    uw1_prod = pipeline_files.analysis_bg_subtracted_images[
        epoch_path, SwiftFilter.uw1, stacking_method
    ]
    uw1_prod.load_product()
    uw1_sum = uw1_prod.data_product.data

    uvv_prod = pipeline_files.analysis_bg_subtracted_images[
        epoch_path, SwiftFilter.uvv, stacking_method
    ]
    uvv_prod.load_product()
    uvv_sum = uvv_prod.data_product.data

    if pipeline_files.analysis_background_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.analysis_background_products!"
        )
        return
    bg_prod = pipeline_files.analysis_background_products[epoch_path]
    if not bg_prod.product_path.exists():
        print(
            f"The background analysis for {epoch_path.stem} has not been done! Exiting."
        )
        return
    bg_prod.load_product()
    bgresults = yaml_dict_to_background_analysis(bg_prod.data_product)
    uw1_bg = bgresults[SwiftFilter.uw1]
    uvv_bg = bgresults[SwiftFilter.uvv]

    rpsp = RadialProfileSelectionPlot(
        epoch=epoch, uw1_img=uw1_sum, uw1_bg=uw1_bg, uvv_img=uvv_sum, uvv_bg=uvv_bg
    )
    rpsp.show()


def main():
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    # load the config
    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    # swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    # available_functions = ["sun_direction", "profile_test", "exit"]
    # selection = available_functions[get_selection(available_functions)]
    # print(selection)
    # if selection == "sun_direction":
    #     find_sun_direction(swift_data=swift_data, pipeline_files=pipeline_files)
    # elif selection == "profile_test":
    #     profile_test_plot(pipeline_files=pipeline_files)
    # else:
    #     return

    profile_test_plot(pipeline_files=pipeline_files)


if __name__ == "__main__":
    sys.exit(main())
