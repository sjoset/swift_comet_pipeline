#!/usr/bin/env python3

import os
import pathlib
import sys
import glob
import numpy as np
import pandas as pd
import logging as log
from itertools import product

from astropy.io import fits

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)

from configs import read_swift_pipeline_config, read_swift_project_config
from epochs import read_epoch
from reddening_correction import DustReddeningPercent
from swift_types import (
    SwiftFilter,
    StackingMethod,
    filter_to_file_string,
    SwiftUVOTImage,
)

from fluorescence_OH import flux_OH_to_num_OH
from flux_OH import OH_flux_from_count_rate, beta_parameter

from num_OH_to_Q import num_OH_to_Q_vectorial
from user_input import get_selection, get_float

from determine_background import (
    BackgroundDeterminationMethod,
    BackgroundResult,
    determine_background,
)
from comet_signal import CometPhotometryMethod, comet_photometry

__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config", nargs=1, help="Filename of project config"
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


def show_fits_sum_and_median_scaled(
    image_sum,
    image_median,
    comet_aperture_radius,
    comet_center_x,
    comet_center_y,
    bg_aperture_x,
    bg_aperture_y,
    bg_aperture_radius,
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    zscale = ZScaleInterval()
    vmin1, vmax1 = zscale.get_limits(image_sum)
    vmin2, vmax2 = zscale.get_limits(image_median)

    im1 = ax1.imshow(image_sum, vmin=vmin1, vmax=vmax1)
    im2 = ax2.imshow(image_median, vmin=vmin2, vmax=vmax2)

    fig.colorbar(im1)
    fig.colorbar(im2)

    image_center_row = int(np.floor(image_sum.shape[0] / 2))
    image_center_col = int(np.floor(image_sum.shape[1] / 2))
    # print(f"Image center: {image_center_col}, {image_center_row}")
    ax1.add_patch(
        plt.Circle(
            (comet_center_x, comet_center_y),
            radius=comet_aperture_radius,
            fill=False,
        )
    )
    ax1.axvline(image_center_col, color="w", alpha=0.3)
    ax1.axhline(image_center_row, color="w", alpha=0.3)

    ax2.add_patch(
        plt.Circle(
            (bg_aperture_x, bg_aperture_y),
            radius=bg_aperture_radius,
            fill=False,
        )
    )

    plt.show()


def show_fits_subtracted(uw1, uvv, beta):
    # adjust for the different count rates between filters
    beta *= 6.0

    dust_map = -(uw1 - beta * uvv)

    # dust_scaled = np.log10(np.clip(dust_map, 0, None) + eps)
    dust_scaled = np.log10(dust_map)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(dust_scaled)

    im1 = ax1.imshow(dust_scaled, vmin=vmin, vmax=vmax)
    # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    image_center_row = int(np.floor(uw1.shape[0] / 2))
    image_center_col = int(np.floor(uw1.shape[1] / 2))
    ax1.axvline(image_center_col, color="b", alpha=0.2)
    ax1.axhline(image_center_row, color="b", alpha=0.2)

    # hdu = fits.PrimaryHDU(dust_subtracted)
    # hdu.writeto("subtracted.fits", overwrite=True)
    plt.show()


def show_background_subtraction(
    before,
    after,
    # comet_aperture_radius,
    # comet_center_x,
    # comet_center_y,
    bg_aperture_x,
    bg_aperture_y,
    bg_aperture_radius,
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    zscale = ZScaleInterval()
    vmin1, vmax1 = zscale.get_limits(before)
    # vmin2, vmax2 = zscale.get_limits(img2)

    im1 = ax1.imshow(before, vmin=vmin1, vmax=vmax1)
    im2 = ax2.imshow(after, vmin=vmin1, vmax=vmax1)

    fig.colorbar(im1)
    fig.colorbar(im2)

    image_center_row = int(np.floor(before.shape[0] / 2))
    image_center_col = int(np.floor(before.shape[1] / 2))
    print(f"Image center: {image_center_col}, {image_center_row}")
    # ax1.add_patch(
    #     plt.Circle(
    #         (comet_center_x, comet_center_y),
    #         radius=comet_aperture_radius,
    #         fill=False,
    #     )
    # )
    ax1.axvline(image_center_col, color="w", alpha=0.3)
    ax1.axhline(image_center_row, color="w", alpha=0.3)

    ax2.add_patch(
        plt.Circle(
            (bg_aperture_x, bg_aperture_y),
            radius=bg_aperture_radius,
            fill=False,
        )
    )

    plt.show()


# def get_aperture_photometry_results(stacked_images) -> dict:
#     comet_aperture_radius = get_float("Comet aperture radius: ")
#     bg_aperture_x = get_float("Background aperture x: ")
#     bg_aperture_y = get_float("Background aperture y: ")
#     bg_aperture_radius = get_float("Background aperture radius: ")
#
#     # collect aperture results for uw1 and uvv filters from the stacked images
#     aperture_results = {}
#     for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
#         summed_image = stacked_images[(filter_type, StackingMethod.summation)]
#         median_image = stacked_images[(filter_type, StackingMethod.median)]
#
#         aperture_results[filter_type] = do_aperture_photometry(
#             stacked_sum=summed_image,
#             stacked_median=median_image,
#             comet_aperture_radius=comet_aperture_radius,
#             bg_aperture_radius=bg_aperture_radius,
#             bg_aperture_x=bg_aperture_x,
#             bg_aperture_y=bg_aperture_y,
#         )
#
#         show_fits_sum_and_median_scaled(
#             summed_image.stacked_image,
#             median_image.stacked_image,
#             comet_aperture_radius=comet_aperture_radius,
#             comet_center_x=aperture_results[filter_type].comet_aperture.positions[0],
#             comet_center_y=aperture_results[filter_type].comet_aperture.positions[1],
#             bg_aperture_x=bg_aperture_x,
#             bg_aperture_y=bg_aperture_y,
#             bg_aperture_radius=bg_aperture_radius,
#         )
#
#     return aperture_results


def stacked_image_from_epoch_path(
    epoch_path: pathlib.Path, filter_type: SwiftFilter, stacking_method: StackingMethod
) -> SwiftUVOTImage:
    epoch_name = epoch_path.stem
    # TODO: update this once we make building this string a function
    fits_filename = (
        f"{epoch_name}_{filter_to_file_string(filter_type)}_{stacking_method}.fits"
    )
    fits_path = epoch_path.parent / pathlib.Path(fits_filename)
    print(f"Loading fits file {fits_path} ...")
    image_data = fits.getdata(fits_path)
    return image_data  # type: ignore


def select_stacked_epoch(stack_dir_path: pathlib.Path) -> pathlib.Path:
    glob_pattern = str(stack_dir_path / pathlib.Path("*.parquet"))
    epoch_filename_list = sorted(glob.glob(glob_pattern))
    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])

    return epoch_path


def get_background(img: SwiftUVOTImage) -> BackgroundResult:
    # TODO: menu here for type of BG
    bg_cr = determine_background(
        img=img,
        background_method=BackgroundDeterminationMethod.gui_manual_aperture,
    )

    return bg_cr


# def get_background(img: SwiftUVOTImage) -> BackgroundResult:
#     # bg_ap_x = 1399.0
#     # bg_ap_y = 1203.0
#     # bg_r = 90.0
#     bg_ap_x = get_float("Background aperture x: ")
#     bg_ap_y = get_float("Background aperture y: ")
#     bg_r = get_float("Background aperture radius: ")
#
#     bg_cr = determine_background(
#         img=img,
#         background_method=BackgroundDeterminationMethod.manual_aperture,
#         aperture_x=bg_ap_x,
#         aperture_y=bg_ap_y,
#         aperture_radius=bg_r,
#     )
#
#     show_background_subtraction(
#         before=img,
#         after=img - bg_cr.count_rate_per_pixel,
#         # comet_aperture_radius=c_r,
#         # comet_center_x=img_center_col,
#         # comet_center_y=img_center_row,
#         bg_aperture_x=bg_ap_x,
#         bg_aperture_y=bg_ap_y,
#         bg_aperture_radius=bg_r,
#     )
#
#     return bg_cr


def do_comet_photometry(
    img: SwiftUVOTImage,
    aperture_radius: float,
):
    img_center_row = int(np.floor(img.shape[0] / 2))
    img_center_col = int(np.floor(img.shape[1] / 2))

    cr = comet_photometry(
        img=img,
        photometry_method=CometPhotometryMethod.manual_aperture,
        aperture_x=img_center_col,
        aperture_y=img_center_row,
        aperture_radius=aperture_radius,
    )

    return cr


def main():
    args = process_args()

    swift_pipeline_config = read_swift_pipeline_config()
    if swift_pipeline_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    stack_dir_path = swift_project_config.stack_dir_path
    if stack_dir_path is None:
        print(f"Could not find stack_dir_path in {swift_project_config_path}, exiting.")
        return

    epoch_path = select_stacked_epoch(stack_dir_path)
    epoch = read_epoch(epoch_path=epoch_path)
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    uw1_sum = stacked_image_from_epoch_path(
        epoch_path=epoch_path,
        filter_type=SwiftFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    # uw1_median = stacked_image_from_epoch_path(
    #     epoch_path=epoch_path,
    #     filter_type=SwiftFilter.uw1,
    #     stacking_method=StackingMethod.median,
    # )
    uvv_sum = stacked_image_from_epoch_path(
        epoch_path=epoch_path,
        filter_type=SwiftFilter.uvv,
        stacking_method=StackingMethod.summation,
    )
    # uvv_median = stacked_image_from_epoch_path(
    #     epoch_path=epoch_path,
    #     filter_type=SwiftFilter.uvv,
    #     stacking_method=StackingMethod.median,
    # )

    # bguw1 = get_background(uw1_median, SwiftFilter.uw1)
    bguw1 = get_background(uw1_sum)
    # bguvv = get_background(uvv_median, SwiftFilter.uvv)
    bguvv = get_background(uvv_sum)
    print(f"Background count rate for uw1: {bguw1}")
    print(f"Background count rate for uvv: {bguvv}")

    # uw1 = np.clip(uw1_sum - bguw1.count_rate_per_pixel, 0, None)
    uw1 = uw1_sum - bguw1.count_rate_per_pixel
    # uvv = np.clip(uvv_sum - bguvv.count_rate_per_pixel, 0, None)
    uvv = uvv_sum - bguvv.count_rate_per_pixel

    aperture_radii = np.linspace(1, 100, num=1000)
    rednesses = [10]
    dust_redness_list = list(
        # map(lambda x: DustReddeningPercent(x), [0, 5, 10, 15, 20, 25])
        map(lambda x: DustReddeningPercent(x), rednesses)
    )
    redness_to_beta = {x.reddening: beta_parameter(x) for x in dust_redness_list}
    print(dust_redness_list)
    print(redness_to_beta)

    count_uw1 = []
    count_uvv = []
    rs = []
    reds = []
    qs = []
    flux = []
    oh = []
    for ap_r, redness in product(aperture_radii, dust_redness_list):
        rs.append(ap_r)
        reds.append(redness)

        cuw1 = do_comet_photometry(img=uw1, aperture_radius=ap_r)
        cuvv = do_comet_photometry(img=uvv, aperture_radius=ap_r)
        count_uw1.append(cuw1)
        count_uvv.append(cuvv)

        flux_OH = OH_flux_from_count_rate(
            count_rate_uw1=cuw1,
            count_rate_uvv=cuvv,
            beta=redness_to_beta[redness.reddening],
        )
        flux.append(flux_OH)

        num_oh = flux_OH_to_num_OH(
            flux_OH=flux_OH,
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta,
        )
        oh.append(num_oh)

        q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
        qs.append(q)

    df = pd.DataFrame(
        {
            "aperture_radius": rs,
            "dust_redness": reds,
            "counts_uw1": count_uw1,
            "counts_uvv": count_uvv,
            "flux_OH": flux,
            "num_OH": oh,
            "Q_H2O": qs,
        }
    )

    qmask = df["Q_H2O"] > 0
    print(df[qmask])
    # print(np.diff(df[qmask]["Q_H2O"]))

    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].set_yscale("log")
    # df.plot.line(x="aperture_radius", y="Q_H2O", subplots=True, ax=axs[0])
    # df.plot.line(x="aperture_radius", y="Q_H2O", subplots=True, ax=axs[1])
    # plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=2)
    # axs[0].set_yscale("log")
    df.plot.line(x="aperture_radius", y="counts_uw1", subplots=True, ax=axs[0])
    df.plot.line(x="aperture_radius", y="counts_uvv", subplots=True, ax=axs[1])
    plt.show()

    # show_fits_subtracted(uw1=uw1_sum, uvv=uvv_sum, beta=beta)

    # img_center_row = int(np.floor(uw1_sum.shape[0] / 2))
    # img_center_col = int(np.floor(uw1_sum.shape[1] / 2))
    # c_r = 30.0
    # cuw1 = comet_photometry(
    #     img=uw1_sum,
    #     filter_type=SwiftFilter.uw1,
    #     stacking_method=StackingMethod.summation,
    #     photometry_method=CometPhotometryMethod.manual_aperture,
    #     aperture_x=img_center_col,
    #     aperture_y=img_center_row,
    #     aperture_radius=c_r,
    # )
    # cuvv = comet_photometry(
    #     img=uvv_sum,
    #     filter_type=SwiftFilter.uvv,
    #     stacking_method=StackingMethod.summation,
    #     photometry_method=CometPhotometryMethod.manual_aperture,
    #     aperture_x=img_center_col,
    #     aperture_y=img_center_row,
    #     aperture_radius=c_r,
    # )
    # print(f"Count rate in uw1 aperture without background subtraction: {cuw1}")
    # print(f"Count rate in uvv aperture without background subtraction: {cuvv}")
    #
    # uw1cr = comet_photometry(
    #     img=uw1_sum - bguw1.count_rate_per_pixel,
    #     filter_type=SwiftFilter.uw1,
    #     stacking_method=StackingMethod.summation,
    #     photometry_method=CometPhotometryMethod.manual_aperture,
    #     aperture_x=img_center_col,
    #     aperture_y=img_center_row,
    #     aperture_radius=c_r,
    # )
    # uvvcr = comet_photometry(
    #     img=uvv_sum - bguvv.count_rate_per_pixel,
    #     filter_type=SwiftFilter.uvv,
    #     stacking_method=StackingMethod.summation,
    #     photometry_method=CometPhotometryMethod.manual_aperture,
    #     aperture_x=img_center_col,
    #     aperture_y=img_center_row,
    #     aperture_radius=c_r,
    # )
    # print(f"Count rate in uw1 aperture with background subtraction: {uw1cr}")
    # print(f"Count rate in uvv aperture with background subtraction: {uvvcr}")

    # show_background_subtraction(
    #     before=uw1_sum,
    #     after=uw1_sum - bguw1.count_rate_per_pixel,
    #     comet_aperture_radius=c_r,
    #     comet_center_x=img_center_col,
    #     comet_center_y=img_center_row,
    #     bg_aperture_x=bg_ap_x,
    #     bg_aperture_y=bg_ap_y,
    #     bg_aperture_radius=bg_r,
    # )
    # show_background_subtraction(
    #     before=uvv_sum,
    #     after=uvv_sum - bguvv.count_rate_per_pixel,
    #     comet_aperture_radius=c_r,
    #     comet_center_x=img_center_col,
    #     comet_center_y=img_center_row,
    #     bg_aperture_x=bg_ap_x,
    #     bg_aperture_y=bg_ap_y,
    #     bg_aperture_radius=bg_r,
    # )


if __name__ == "__main__":
    sys.exit(main())
