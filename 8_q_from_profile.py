#!/usr/bin/env python3

import os
import pathlib
import sys
import glob

# import yaml
import numpy as np

# import pandas as pd
import logging as log

# from typing import List
# from itertools import product

from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus

# from astropy import modeling

# from scipy.optimize import curve_fit

# import scipy.interpolate as interp

# from tqdm import tqdm

from argparse import ArgumentParser

# import matplotlib.pyplot as plt

# from astropy.visualization import ZScaleInterval

from configs import read_swift_project_config

# from error_propogation import ValueAndStandardDev
from pipeline_files import PipelineFiles

# from reddening_correction import DustReddeningPercent
from swift_filter import SwiftFilter, filter_to_file_string
from stacking import StackingMethod
from uvot_image import SwiftUVOTImage, get_uvot_image_center

# from fluorescence_OH import flux_OH_to_num_OH
# from flux_OH import OH_flux_from_count_rate, beta_parameter
# from num_OH_to_Q import num_OH_to_Q_vectorial
from tui import get_selection, stacked_epoch_menu
from determine_background import (
    # BackgroundDeterminationMethod,
    # BackgroundResult,
    # determine_background,
    yaml_dict_to_background_analysis,
)
from comet_signal import (
    CometCenterFindingMethod,
    find_comet_center,
)
from comet_profile import (
    CometProfile,
    count_rate_from_comet_profile,
    count_rate_from_comet_radial_profile,
    estimate_comet_radius_at_angle,
    extract_comet_radial_profile,
    fit_comet_profile_gaussian,
    plot_fitted_profile,
)

# from plateau_detect import plateau_detect
# from epochs import Epoch
# from count_rate import CountRate, CountRatePerPixel, magnitude_from_count_rate


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
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


# def show_fits_subtracted(uw1, uvv, beta):
#     # adjust for the different count rates between filters
#     beta *= 6.0
#
#     dust_map = -(uw1 - beta * uvv)
#
#     # dust_scaled = np.log10(np.clip(dust_map, 0, None) + eps)
#     dust_scaled = np.log10(dust_map)
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#
#     zscale = ZScaleInterval()
#     vmin, vmax = zscale.get_limits(dust_scaled)
#
#     im1 = ax1.imshow(dust_scaled, vmin=vmin, vmax=vmax)
#     # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
#     fig.colorbar(im1)
#
#     image_center_row = int(np.floor(uw1.shape[0] / 2))
#     image_center_col = int(np.floor(uw1.shape[1] / 2))
#     ax1.axvline(image_center_col, color="b", alpha=0.2)
#     ax1.axhline(image_center_row, color="b", alpha=0.2)
#
#     # hdu = fits.PrimaryHDU(dust_subtracted)
#     # hdu.writeto("subtracted.fits", overwrite=True)
#     plt.show()


# def show_centers(img, cs: List[PixelCoord]):
#     # img_scaled = np.log10(img)
#     img_scaled = img
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#
#     pix_center = get_uvot_image_center(img)
#     ax1.add_patch(
#         plt.Circle(
#             (pix_center.x, pix_center.y),
#             radius=30,
#             fill=False,
#         )
#     )
#
#     zscale = ZScaleInterval()
#     vmin, vmax = zscale.get_limits(img_scaled)
#
#     im1 = ax1.imshow(img_scaled, vmin=vmin, vmax=vmax)
#     fig.colorbar(im1)
#
#     for c in cs:
#         line_color = next(ax1._get_lines.prop_cycler)["color"]
#         ax1.axvline(c.x, alpha=0.7, color=line_color)
#         ax1.axhline(c.y, alpha=0.9, color=line_color)
#
#     plt.show()


def select_stacked_epoch(stack_dir_path: pathlib.Path) -> pathlib.Path:
    glob_pattern = str(stack_dir_path / pathlib.Path("*.parquet"))
    epoch_filename_list = sorted(glob.glob(glob_pattern))
    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])

    return epoch_path


# def get_background(img: SwiftUVOTImage, filter_type: SwiftFilter) -> BackgroundResult:
#     # TODO: menu here for type of BG method
#     bg_cr = determine_background(
#         img=img,
#         background_method=BackgroundDeterminationMethod.gui_manual_aperture,
#         filter_type=filter_type,
#     )
#
#     return bg_cr


def compare_comet_center_methods(uw1: SwiftUVOTImage, uvv: SwiftUVOTImage):
    # TODO: uvv tends to pick up the dust tail so we might expect some difference depending on the method of center detection,
    # so maybe we just use the uw1 and assume it's less likely to have a tail to scramble the center-finding

    peaks = {}
    imgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        print(f"Determining center of comet for {filter_to_file_string(filter_type)}:")
        pix_center = get_uvot_image_center(img=imgs[filter_type])
        search_ap = CircularAperture((pix_center.x, pix_center.y), r=30)
        pixel_center = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.pixel_center,
            search_aperture=search_ap,
        )
        centroid = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.aperture_centroid,
            search_aperture=search_ap,
        )
        peak = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.aperture_peak,
            search_aperture=search_ap,
        )
        print("\tBy image center: ", pixel_center)
        print(
            "\tBy centroid (center of mass) in aperture radius 30 at image center: ",
            centroid,
        )
        print("\tBy peak value in aperture radius 30 at image center: ", peak)

        peaks[filter_type] = peak

    xdist = peaks[SwiftFilter.uw1].x - peaks[SwiftFilter.uvv].x
    ydist = peaks[SwiftFilter.uw1].y - peaks[SwiftFilter.uvv].y
    dist = np.sqrt(xdist**2 + ydist**2)
    if dist > np.sqrt(2.0):
        print(
            f"Comet peaks in uw1 and uvv are separated by {dist} pixels! Fitting might suffer."
        )

    # show_centers(uw1, [pixel_center, centroid, peak])  # pyright: ignore


# def fit_inverse_r(img: SwiftUVOTImage) -> None:
#     profile_radius = 40
#
#     pix_center = get_uvot_image_center(img=img)
#     search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
#     peak = find_comet_center(
#         img=img,
#         method=CometCenterFindingMethod.aperture_peak,
#         search_aperture=search_aperture,
#     )
#     comet_profile = count_rate_profile(
#         img=img,
#         comet_center=peak,
#         theta=0,
#         r=profile_radius,
#     )
#
#     mask = comet_profile.distances_from_center > 0
#     rs = np.log10(comet_profile.distances_from_center[mask])
#     pix = comet_profile.pixel_values[mask]
#
#     def log_dust_profile(r, a, b):
#         return a * r + b
#
#     dust_fit = curve_fit(
#         log_dust_profile,
#         rs,
#         pix,
#         [-1, 0],
#     )
#
#     a_fit = dust_fit[0][0]
#     b_fit = dust_fit[0][1]
#
#     print(f"{a_fit=}, {b_fit=}")
#
#     plt.plot(
#         rs,
#         log_dust_profile(rs, a_fit, b_fit),
#     )
#     plt.plot(
#         np.log10(np.abs(comet_profile.distances_from_center)),
#         comet_profile.pixel_values,
#     )
#     plt.show()


def test_circular_aperture_vs_donut_stack(img: SwiftUVOTImage) -> None:
    """
    Computes the total signal from a large aperture against concentric annulus apertures

    If we want to compute signal as a function of aperture radius, we can either re-calculate an entirely new aperture
    at each r, or keep a running total of annulus results from from r=0 up to r-1 pixels and add the results from a thin annulus
    at r=r

    This examines the difference between the two approaches to make sure the "donut stack" results are an accurate signal count
    """

    profile_radius = 30
    num_donuts = 10
    pix_center = get_uvot_image_center(img=img)
    search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    peak = find_comet_center(
        img=img,
        method=CometCenterFindingMethod.aperture_peak,
        search_aperture=search_aperture,
    )

    inner_rs, r_step = np.linspace(
        0.001,
        profile_radius - (profile_radius / num_donuts),
        num=num_donuts,
        endpoint=True,
        retstep=True,
    )
    # outer_rs = inner_rs + r_step
    # mid_rs = (outer_rs + inner_rs) / 2

    ap_stats_list = []
    for inner_r in inner_rs:
        outer_r = inner_r + r_step
        ap = CircularAnnulus((peak.x, peak.y), r_in=inner_r, r_out=outer_r)
        ap_stats_list.append(ApertureStats(img, ap))

    total_signal = np.sum([x.sum for x in ap_stats_list])

    total_aperture = CircularAperture((peak.x, peak.y), r=profile_radius)
    total_stats = ApertureStats(img, total_aperture)
    print(
        f"From circular aperture: {total_stats.sum} ({total_stats.sum * 100/total_signal})% of annulus stack signal"
    )

    pass


# def stacked_epoch_menu(pipeline_files: PipelineFiles) -> Optional[pathlib.Path]:
#     if pipeline_files.epoch_products is None:
#         return None
#
#     epoch_paths = [x.product_path for x in pipeline_files.epoch_products]
#     stacked_epoch_paths = [
#         pipeline_files.stacked_epoch_products[x].product_path for x in epoch_paths  # type: ignore
#     ]
#
#     # filter epochs out of the list if we haven't stacked it by seeing if the stacked_epoch_path exists or not
#     filtered_epochs = list(
#         filter(lambda x: x[1].exists(), zip(epoch_paths, stacked_epoch_paths))
#     )
#
#     selectable_epochs = [x[0] for x in filtered_epochs]
#     if len(selectable_epochs) == 0:
#         return None
#     selection = get_selection(selectable_epochs)
#     return selectable_epochs[selection]


def main():
    args = process_args()

    # load the config
    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    # select the epoch we want to process
    epoch_path = stacked_epoch_menu(pipeline_files=pipeline_files)
    if epoch_path is None:
        print("No stacked images found! Exiting.")
        return 1
    if pipeline_files.stacked_epoch_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.stacked_epoch_products!"
        )
        return 1

    # load the epoch database
    pipeline_files.stacked_epoch_products[epoch_path].load_product()
    epoch = pipeline_files.stacked_epoch_products[epoch_path].data_product
    print(
        f"Starting analysis of {epoch_path.stem}: observation at {np.mean(epoch.HELIO)} AU"
    )

    # more sanity checks
    if pipeline_files.stacked_image_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.stacked_image_products!"
        )
        return 1
    if pipeline_files.analysis_bg_subtracted_images is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.analysis_bg_subtracted_images!"
        )
        return 1

    stacking_method = StackingMethod.summation

    # load the background-subtracted images into 'uw1' and 'uvv' as numpy arrays
    uw1_prod = pipeline_files.analysis_bg_subtracted_images[
        epoch_path, SwiftFilter.uw1, stacking_method
    ]
    uvv_prod = pipeline_files.analysis_bg_subtracted_images[
        epoch_path, SwiftFilter.uvv, stacking_method
    ]
    if not uw1_prod.product_path.exists() or not uvv_prod.product_path.exists():
        print(
            f"The background-subtracted images for {epoch_path.stem} need to be generated! Exiting."
        )
        return 1
    uw1_prod.load_product()
    uvv_prod.load_product()
    uw1 = uw1_prod.data_product.data
    uvv = uvv_prod.data_product.data

    # load the background analysis values and uncertainties for error propogation
    if pipeline_files.analysis_background_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.analysis_background_products!"
        )
        return 1
    bg_prod = pipeline_files.analysis_background_products[epoch_path]
    if not bg_prod.product_path.exists():
        print(
            f"The background analysis for {epoch_path.stem} has not been done! Exiting."
        )
        return 1
    bg_prod.load_product()
    bgresults = yaml_dict_to_background_analysis(bg_prod.data_product)

    profile_radius = 20
    pix_center = get_uvot_image_center(img=uvv)
    peak = pix_center

    # TODO: check if uw1 and uvv peak are the same, then use peak, otherwise, image center
    # search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    # peak = find_comet_center(
    #     img=uw1,
    #     method=CometCenterFindingMethod.aperture_peak,
    #     search_aperture=search_aperture,
    # )

    imgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
    radius_estimation = {}

    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        # peak = pix_center
        # test_circular_aperture_vs_donut_stack(img=imgs[filter_type])

        # count_list = []
        # ecr_list = []
        # thetas = np.linspace(0, np.pi, num=50, endpoint=False)
        # for theta in thetas:
        #     comet_profile = count_rate_profile(
        #         img=imgs[filter_type],
        #         comet_center=peak,
        #         theta=theta,
        #         r=profile_radius,
        #     )
        #     ccr = count_rate_from_count_rate_profile(
        #         comet_profile, bgresults[filter_type].count_rate_per_pixel
        #     )
        #     # print(f"From profile at {theta=}: {ccr}")
        #     count_list.append(ccr)
        #
        #     ecr = estimate_comet_radius_at_angle(
        #         img=imgs[filter_type],
        #         comet_center=peak,
        #         radius_guess=profile_radius,
        #         theta=theta,
        #     )
        #     # print(f"Estimated comet radius: {ecr}")
        #     ecr_list.append(ecr)
        #
        # print("Using 50 cuts of comet profile at different angles:")
        # cl_mean = np.mean([x.value for x in count_list])
        # cl_median = np.median([x.value for x in count_list])
        # cl_std = np.std([x.value for x in count_list])
        # print(
        #     f"Count rate summary: avg = {cl_mean}, median = {cl_median}, std = {cl_std}"
        # )
        # print(
        #     f"Estimated comet radius: avg = {np.mean(ecr_list)}, median = {np.median(ecr_list)}, std = {np.std(ecr_list)}"
        # )
        # print("")
        #
        # # cfe_ap = CircularAperture((peak.x, peak.y), np.mean(ecr_list))
        # # cfe_stats = ApertureStats(imgs[filter_type], cfe_ap)
        # # print(f"Counts in aperture of average radius from estimation: {cfe_stats.sum}")
        #
        # radius_estimation[filter_type] = np.mean(ecr_list)

        # theta_list, radii_list = estimate_comet_radius_by_angle(
        #     img=imgs[filter_type],
        #     comet_center=peak,
        #     radius_guess=30,
        # )
        #
        # rl_ap = CircularAperture((peak.x, peak.y), np.mean(radii_list))
        # rl_stats = ApertureStats(imgs[filter_type], rl_ap)
        # print(f"Counts in aperture of average radius list: {rl_stats.sum}")

        # just plot along specified theta, mark 4-sigma threshold
        comet_radial_profile = extract_comet_radial_profile(
            img=imgs[filter_type],
            comet_center=peak,
            theta=0.0,
            r=profile_radius,
        )
        comet_profile = CometProfile.from_radial_profile(comet_radial_profile)
        print(
            count_rate_from_comet_radial_profile(
                comet_radial_profile, bg=bgresults[filter_type].count_rate_per_pixel
            )
        )
        print(
            count_rate_from_comet_profile(
                comet_profile=comet_profile,
                bg=bgresults[filter_type].count_rate_per_pixel,
            )
        )
        print(
            estimate_comet_radius_at_angle(
                img=imgs[filter_type],
                comet_center=peak,
                radius_guess=profile_radius,
                theta=0.0,
            )
        )
        fitted_model = fit_comet_profile_gaussian(comet_profile=comet_profile)
        plot_fitted_profile(
            comet_profile=comet_profile,
            fitted_model=fitted_model,
            sigma_threshold=4.0,
            plot_title=f"Comet profile along theta = 0, with estimated aperture radius for filter {filter_to_file_string(filter_type)}",
        )

    # max_radius = np.max(
    #     [radius_estimation[SwiftFilter.uw1], radius_estimation[SwiftFilter.uvv]]
    # )
    # print(f"Taking the radius to be {max_radius} for both filters")
    # helio_r_au = np.mean(epoch.HELIO)
    # helio_v_kms = np.mean(epoch.HELIO_V)
    # delta = np.mean(epoch.OBS_DIS)
    #
    # uw1cr = comet_manual_aperture(
    #     imgs[SwiftFilter.uw1],
    #     aperture_x=peak.x,
    #     aperture_y=peak.y,
    #     aperture_radius=max_radius,
    #     bg=bgresults[SwiftFilter.uw1].count_rate_per_pixel,
    # )
    # uvvcr = comet_manual_aperture(
    #     imgs[SwiftFilter.uvv],
    #     aperture_x=peak.x,
    #     aperture_y=peak.y,
    #     aperture_radius=max_radius,
    #     bg=bgresults[SwiftFilter.uvv].count_rate_per_pixel,
    # )
    #
    # correction_factors = np.linspace(1.0, 3.0, num=50, endpoint=True)
    # correction_factors = np.append(correction_factors, [10.0, 100.0])
    # for correction_factor in correction_factors:
    #     flux_OH = OH_flux_from_count_rate(
    #         uw1=ValueAndStandardDev(
    #             value=uw1cr.value * correction_factor, sigma=uw1cr.sigma
    #         ),
    #         uvv=uvvcr,
    #         beta=beta_parameter(DustReddeningPercent(reddening=10)),
    #     )
    #
    #     num_oh = flux_OH_to_num_OH(
    #         flux_OH=flux_OH,
    #         helio_r_au=helio_r_au,
    #         helio_v_kms=helio_v_kms,
    #         delta_au=delta,
    #     )
    #
    #     q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
    #
    #     print(
    #         f"Correction factor {correction_factor:7.6f}\t\tQ from best guess: {q.value:7.6e}"
    #     )

    # xs_list = []
    # ys_list = []
    # zs_list = []
    #
    # profile_radius = 30
    # # angles = [3 * np.pi / 2, np.pi / 2, np.pi / 4]
    # angles = np.linspace(0, np.pi, 50)
    # angles = angles[:-1]
    # for theta in angles:
    #     xs, ys, prof = extract_profile(
    #         img=uvv, r=profile_radius, theta=theta, plot_profile=False
    #     )
    #     xs_list.extend(xs)
    #     ys_list.extend(ys)
    #     zs_list.extend(prof)
    #
    # pix_center = get_uvot_image_center(img=uvv)
    # search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    #
    # comet_center = find_comet_center(
    #     img=uvv,
    #     method=CometCenterFindingMethod.aperture_peak,
    #     search_aperture=search_aperture,
    # )
    #
    # mx, my = np.meshgrid(
    #     np.linspace(
    #         comet_center[0] - profile_radius / 2, comet_center[0] + profile_radius / 2
    #     ),
    #     np.linspace(
    #         comet_center[1] - profile_radius / 2, comet_center[1] + profile_radius / 2
    #     ),
    # )
    # mz = interp.griddata((xs_list, ys_list), zs_list, (mx, my), method="linear")
    #
    # ax1.plot_surface(mx, my, mz, cmap="magma")
    #
    # # ax.plot_trisurf(xs_list, ys_list, zs_list, cmap="magma")
    # ax2.contourf(mx, my, mz, cmap="magma")
    #
    # plt.show()

    # next step in pipeline should be to decide redness and aperture radius?

    # decide radius --> inform vectorial model about extent of grid?

    # dust profile + column density fitting? --> redness


if __name__ == "__main__":
    sys.exit(main())
