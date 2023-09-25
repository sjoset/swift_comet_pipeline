#!/usr/bin/env python3

# TODO: rename this to 6_analyze_stack.py

import os
import pathlib
import sys
import glob

# import yaml
import numpy as np
import pandas as pd
import logging as log
from typing import List
from itertools import product

from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus

from astropy.io import fits

# from astropy import modeling

from scipy.optimize import curve_fit

# import scipy.interpolate as interp

from tqdm import tqdm

from argparse import ArgumentParser
import matplotlib.pyplot as plt

# from astropy.visualization import ZScaleInterval

from configs import read_swift_project_config
from epochs import read_epoch
from pipeline_files import PipelineFiles

from reddening_correction import DustReddeningPercent
from swift_filter import SwiftFilter, filter_to_file_string
from stacking import StackingMethod
from uvot_image import SwiftUVOTImage, get_uvot_image_center
from fluorescence_OH import flux_OH_to_num_OH
from flux_OH import OH_flux_from_count_rate, beta_parameter
from num_OH_to_Q import num_OH_to_Q_vectorial
from user_input import get_selection
from determine_background import (
    BackgroundDeterminationMethod,
    BackgroundResult,
    determine_background,
    # read_background_analysis,
    write_background_analysis,
)
from comet_signal import (
    CometCenterFindingMethod,
    comet_manual_aperture,
    estimate_comet_radius_by_angle,
    estimate_comet_radius_at_angle,
    find_comet_center,
    count_rate_profile,
    count_rate_from_count_rate_profile,
    fit_comet_profile_gaussian,
    plot_fitted_profile,
)
from plateau_detect import plateau_detect
from epochs import Epoch
from count_rate import CountRate, CountRatePerPixel, magnitude_from_count_rate


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


def get_background(img: SwiftUVOTImage, filter_type: SwiftFilter) -> BackgroundResult:
    # TODO: menu here for type of BG method
    bg_cr = determine_background(
        img=img,
        background_method=BackgroundDeterminationMethod.gui_manual_aperture,
        filter_type=filter_type,
    )

    return bg_cr


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


def q_vs_aperture_radius(
    epoch: Epoch,
    uw1: SwiftUVOTImage,
    uvv: SwiftUVOTImage,
    dust_rednesses: List[DustReddeningPercent],
    bguw1: CountRatePerPixel,
    bguvv: CountRatePerPixel,
) -> pd.DataFrame:
    # TODO: schema for this dataframe and write it out
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    # TODO: can vectorial model give a better estimate? might be able to borrow the code for grid size here
    radius_km_guess = 1e5
    r_pix = int(radius_km_guess / np.mean(epoch.KM_PER_PIX))
    print(f"Guessing radius of 1e5 km or {r_pix} pixels")

    aperture_radii, r_step = np.linspace(1, 2 * r_pix, num=2 * r_pix, retstep=True)
    redness_to_beta = {x.reddening: beta_parameter(x) for x in dust_rednesses}

    count_uw1 = []
    count_uvv = []
    rs = []
    red_list = []
    qs = []
    flux = []
    oh = []
    progress_bar = tqdm(
        product(aperture_radii, dust_rednesses),
        total=(len(aperture_radii) * len(dust_rednesses)),
    )
    for ap_r, redness in progress_bar:
        # for ap_r, redness in product(aperture_radii, dust_rednesses):
        rs.append(ap_r)
        red_list.append(redness)

        cuw1 = do_comet_photometry_at_img_center(
            img=uw1, aperture_radius=ap_r, bg=bguw1
        )
        cuvv = do_comet_photometry_at_img_center(
            img=uvv, aperture_radius=ap_r, bg=bguvv
        )

        count_uw1.append(cuw1)
        count_uvv.append(cuvv)

        flux_OH = OH_flux_from_count_rate(
            uw1=cuw1,
            uvv=cuvv,
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

    mags_uw1 = [magnitude_from_count_rate(x, SwiftFilter.uw1) for x in count_uw1]
    mags_uvv = [magnitude_from_count_rate(x, SwiftFilter.uvv) for x in count_uvv]
    df = pd.DataFrame(
        {
            "aperture_radius": rs,
            "dust_redness": list(map(lambda x: int(x.reddening), red_list)),
            "counts_uw1": [x.value for x in count_uw1],
            "sigma_counts_uw1": [x.sigma for x in count_uw1],
            "counts_uvv": [x.value for x in count_uvv],
            "sigma_counts_uvv": [x.sigma for x in count_uvv],
            "mag_uw1": [m.value for m in mags_uw1],
            "sigma_mag_uw1": [m.sigma for m in mags_uw1],
            "mag_uvv": [m.value for m in mags_uvv],
            "sigma_mag_uvv": [m.sigma for m in mags_uvv],
            "flux_OH": [f.value for f in flux],
            "sigma_flux_OH": [f.sigma for f in flux],
            "num_OH": [x.value for x in oh],
            "sigma_num_OH": [x.sigma for x in oh],
            "Q_H2O": [q.value for q in qs],
            "sigma_Q_H2O": [q.sigma for q in qs],
        }
    )
    df["snr_uw1"] = df.counts_uw1 / df.sigma_counts_uw1
    df["snr_uvv"] = df.counts_uvv / df.sigma_counts_uvv

    # qmask = df["Q_H2O"] > 0
    # print(df[qmask])

    print("")
    print("Plateau search in uw1 counts:")
    uw1_plateau_list = plateau_detect(
        ys=df.counts_uw1.values,
        xstep=float(r_step),
        smoothing=3,
        threshold=1e-2,
        min_length=5,
    )
    if len(uw1_plateau_list) > 0:
        for p in uw1_plateau_list:
            i = p.begin_index
            j = p.end_index
            print(
                f"Plateau between r = {df.loc[i].aperture_radius} to r = {df.loc[j].aperture_radius}"
            )
            plateau_slice = df[i:j]
            positive_Qs = plateau_slice[plateau_slice.Q_H2O > 0]
            if len(positive_Qs > 0):
                print(f"\tAverage Q_H2O: {np.mean(positive_Qs.Q_H2O)}")
            else:
                print("\tNo positive Q_H2O values found in this plateau")
    else:
        print("No plateaus in uw1 counts detected")

    print("")
    print("Plateau search in uvv counts:")
    uvv_plateau_list = plateau_detect(
        ys=df.counts_uvv.values,
        xstep=float(r_step),
        smoothing=5,
        threshold=5e-3,
        min_length=5,
    )
    if len(uvv_plateau_list) > 0:
        for p in uvv_plateau_list:
            i = p.begin_index
            j = p.end_index
            print(
                f"Plateau between r = {df.loc[i].aperture_radius} to r = {df.loc[j].aperture_radius}"
            )
            plateau_slice = df[i:j]
            positive_Qs = plateau_slice[plateau_slice.Q_H2O > 0]
            if len(positive_Qs > 0):
                # print(positive_Qs)
                print(f"\tAverage Q_H2O: {np.mean(positive_Qs.Q_H2O)}")
            else:
                print("\tNo positive Q_H2O values found in this plateau")
    else:
        print("No plateaus in uvv counts detected")
    print("")

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[2].set_yscale("log")
    df.plot.line(x="aperture_radius", y="counts_uw1", subplots=True, ax=axs[0])
    axs[0].errorbar(df.aperture_radius, df.counts_uw1, df.sigma_counts_uw1)
    df.plot.line(x="aperture_radius", y="counts_uvv", subplots=True, ax=axs[1])
    axs[1].errorbar(df.aperture_radius, df.counts_uvv, df.sigma_counts_uvv)
    df.plot.line(x="aperture_radius", y="Q_H2O", subplots=True, ax=axs[2])
    axs[2].errorbar(df.aperture_radius, df.Q_H2O, df.sigma_Q_H2O)

    for p in uw1_plateau_list:
        i = p.begin_index
        j = p.end_index
        axs[0].axvspan(df.loc[i].aperture_radius, df.loc[j].aperture_radius, color="blue", alpha=0.1)  # type: ignore

    for p in uvv_plateau_list:
        i = p.begin_index
        j = p.end_index
        axs[1].axvspan(df.loc[i].aperture_radius, df.loc[j].aperture_radius, color="orange", alpha=0.1)  # type: ignore

    plt.show()

    return df


def do_comet_photometry_at_img_center(
    img: SwiftUVOTImage, aperture_radius: float, bg: CountRatePerPixel
) -> CountRate:
    pix_center = get_uvot_image_center(img=img)
    ap_x, ap_y = pix_center.x, pix_center.y

    return comet_manual_aperture(
        img=img,
        aperture_x=ap_x,
        aperture_y=ap_y,
        aperture_radius=aperture_radius,
        bg=bg,
    )


def fit_inverse_r(img: SwiftUVOTImage) -> None:
    profile_radius = 40

    pix_center = get_uvot_image_center(img=img)
    search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    peak = find_comet_center(
        img=img,
        method=CometCenterFindingMethod.aperture_peak,
        search_aperture=search_aperture,
    )
    comet_profile = count_rate_profile(
        img=img,
        comet_center=peak,
        theta=0,
        r=profile_radius,
    )

    mask = comet_profile.distances_from_center > 0
    rs = np.log10(comet_profile.distances_from_center[mask])
    pix = comet_profile.pixel_values[mask]

    def log_dust_profile(r, a, b):
        return a * r + b

    dust_fit = curve_fit(
        log_dust_profile,
        rs,
        pix,
        [-1, 0],
    )

    # print(dust_fit)
    a_fit = dust_fit[0][0]
    b_fit = dust_fit[0][1]

    print(f"{a_fit=}, {b_fit=}")

    plt.plot(
        rs,
        log_dust_profile(rs, a_fit, b_fit),
        # log_dust_profile(comet_profile.distances_from_center, a_fit, b_fit),
    )
    plt.plot(
        np.log10(np.abs(comet_profile.distances_from_center)),
        comet_profile.pixel_values,
    )
    plt.show()


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


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    epoch_path = pathlib.Path(
        pipeline_files.stacked_epoch_paths[
            get_selection(pipeline_files.stacked_epoch_paths)
        ]
    )

    epoch = read_epoch(epoch_path=epoch_path)
    # print(epoch)

    uw1_sum: SwiftUVOTImage = fits.getdata(  # type: ignore
        pipeline_files._get_stacked_image_path(
            epoch_path=epoch_path,
            filter_type=SwiftFilter.uw1,
            stacking_method=StackingMethod.summation,
        )
    )
    uvv_sum: SwiftUVOTImage = fits.getdata(  # type: ignore
        pipeline_files._get_stacked_image_path(
            epoch_path=epoch_path,
            filter_type=SwiftFilter.uvv,
            stacking_method=StackingMethod.summation,
        )
    )

    bguw1_results = get_background(uw1_sum, filter_type=SwiftFilter.uw1)
    bguvv_results = get_background(uvv_sum, filter_type=SwiftFilter.uvv)

    # extract the count rates
    bguw1 = bguw1_results.count_rate_per_pixel
    bguvv = bguvv_results.count_rate_per_pixel
    print("")
    print(f"Background count rate for uw1: {bguw1}")
    print(f"Background count rate for uvv: {bguvv}")

    uw1 = uw1_sum - bguw1.value
    uvv = uvv_sum - bguvv.value

    compare_comet_center_methods(uw1, uvv)

    q_vs_r = q_vs_aperture_radius(
        epoch=epoch,
        uw1=uw1,
        uvv=uvv,
        dust_rednesses=[DustReddeningPercent(reddening=x) for x in [0, 10, 20, 30, 40]],
        bguw1=bguw1,
        bguvv=bguvv,
    )

    fit_inverse_r(uw1)
    fit_inverse_r(uvv)

    profile_radius = 30
    pix_center = get_uvot_image_center(img=uvv)

    # TODO: check if uw1 and uvv peak are the same, then use peak, otherwise, image center
    # search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    # peak = find_comet_center(
    #     img=uw1,
    #     method=CometCenterFindingMethod.aperture_peak,
    #     search_aperture=search_aperture,
    # )

    imgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
    radius_estimation = {}

    bg_crs = {SwiftFilter.uw1: bguw1, SwiftFilter.uvv: bguvv}
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        peak = pix_center
        # test_circular_aperture_vs_donut_stack(img=imgs[filter_type])

        count_list = []
        ecr_list = []
        thetas = np.linspace(0, np.pi, num=50, endpoint=False)
        for theta in thetas:
            comet_profile = count_rate_profile(
                img=imgs[filter_type],
                comet_center=peak,
                theta=theta,
                r=profile_radius,
            )
            ccr = count_rate_from_count_rate_profile(comet_profile, bg_crs[filter_type])
            # print(f"From profile at {theta=}: {ccr}")
            count_list.append(ccr)

            ecr = estimate_comet_radius_at_angle(
                img=imgs[filter_type],
                comet_center=peak,
                radius_guess=profile_radius,
                theta=theta,
            )
            # print(f"Estimated comet radius: {ecr}")
            ecr_list.append(ecr)

        print("Using 50 cuts of comet profile at different angles:")
        cl_mean = np.mean([x.value for x in count_list])
        cl_median = np.median([x.value for x in count_list])
        cl_std = np.std([x.value for x in count_list])
        print(
            f"Count rate summary: avg = {cl_mean}, median = {cl_median}, std = {cl_std}"
        )
        print(
            f"Estimated comet radius: avg = {np.mean(ecr_list)}, median = {np.median(ecr_list)}, std = {np.std(ecr_list)}"
        )
        print("")

        # cfe_ap = CircularAperture((peak.x, peak.y), np.mean(ecr_list))
        # cfe_stats = ApertureStats(imgs[filter_type], cfe_ap)
        # print(f"Counts in aperture of average radius from estimation: {cfe_stats.sum}")

        radius_estimation[filter_type] = np.mean(ecr_list)

        theta_list, radii_list = estimate_comet_radius_by_angle(
            img=imgs[filter_type],
            comet_center=peak,
            radius_guess=30,
        )

        rl_ap = CircularAperture((peak.x, peak.y), np.mean(radii_list))
        rl_stats = ApertureStats(imgs[filter_type], rl_ap)
        print(f"Counts in aperture of average radius list: {rl_stats.sum}")

        comet_profile = count_rate_profile(
            img=imgs[filter_type],
            comet_center=peak,
            theta=0,
            r=profile_radius,
        )
        fitted_model = fit_comet_profile_gaussian(comet_profile=comet_profile)
        plot_fitted_profile(
            comet_profile=comet_profile,
            fitted_model=fitted_model,
            sigma_threshold=4.0,
            plot_title=f"Comet profile along theta = 0, with estimated aperture radius for filter {filter_to_file_string(filter_type)}",
        )

    max_radius = np.max(
        [radius_estimation[SwiftFilter.uw1], radius_estimation[SwiftFilter.uvv]]
    )
    print(f"Taking the radius to be {max_radius} for both filters")
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    uw1cr = comet_manual_aperture(
        imgs[SwiftFilter.uw1],
        aperture_x=peak.x,
        aperture_y=peak.y,
        aperture_radius=max_radius,
        bg=bguw1,
    )
    uvvcr = comet_manual_aperture(
        imgs[SwiftFilter.uvv],
        aperture_x=peak.x,
        aperture_y=peak.y,
        aperture_radius=max_radius,
        bg=bguvv,
    )

    flux_OH = OH_flux_from_count_rate(
        uw1=uw1cr,
        uvv=uvvcr,
        beta=beta_parameter(DustReddeningPercent(reddening=10)),
    )

    num_oh = flux_OH_to_num_OH(
        flux_OH=flux_OH,
        helio_r_au=helio_r_au,
        helio_v_kms=helio_v_kms,
        delta_au=delta,
    )

    q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)

    print(f"Q from best guess: {q}")

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

    ####
    # save the background-subtracted fits
    ####
    bgpath = pipeline_files.get_analysis_background_path(epoch_path=epoch_path)
    write_background_analysis(
        out_path=bgpath,
        method=BackgroundDeterminationMethod.gui_manual_aperture,
        uw1_result=bguw1_results,
        uvv_result=bguvv_results,
    )

    # TODO: do this for both stacking methods or ignore the medians?
    bgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        sub_path = pipeline_files.get_analysis_bg_subtracted_path(
            epoch_path=epoch_path, filter_type=filter_type
        )

        # grab the original stacked image header and copy to background-subtracted fits
        stacked_hdul = fits.open(
            pipeline_files.get_stacked_image_path(
                epoch_path=epoch_path,
                filter_type=filter_type,
                stacking_method=StackingMethod.summation,
            )
        )
        hdr_to_copy = stacked_hdul[1].header  # type: ignore

        bg_hdu = fits.ImageHDU(data=bgs[filter_type])
        bg_hdu.header = hdr_to_copy
        bg_hdu.writeto(sub_path, overwrite=True)

        stacked_hdul.close()

    ####
    # save q vs r
    ####
    q_vs_r_path = pipeline_files.get_analysis_qh2o_vs_aperture_radius_path(
        epoch_path=epoch_path
    )
    q_vs_r.to_csv(q_vs_r_path, index=False)

    # next step in pipeline should be to decide redness and aperture radius?

    # decide radius --> inform vectorial model about extent of grid?

    # dust profile + column density fitting? --> redness


if __name__ == "__main__":
    sys.exit(main())
