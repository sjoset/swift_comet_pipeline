#!/usr/bin/env python3

import os
import pathlib
import sys
import glob
import numpy as np
import pandas as pd
import logging as log
from typing import Tuple, List
from itertools import product

import scipy.interpolate as interp

from photutils.aperture import ApertureStats, CircularAperture

from astropy.io import fits

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from configs import read_swift_pipeline_config, read_swift_project_config
from epochs import read_epoch
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
)
from comet_signal import CometCenterFindingMethod, find_comet_center
from plateau_detect import plateau_detect
from epochs import Epoch


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

    # TODO: this is now a function
    image_center_row = int(np.floor(image_sum.shape[0] / 2))
    image_center_col = int(np.floor(image_sum.shape[1] / 2))
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


def show_centers(img, cs: List[Tuple[float, float]]):
    # img_scaled = np.log10(img)
    img_scaled = img

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    pix_center = get_uvot_image_center(img)
    ax1.add_patch(
        plt.Circle(
            (pix_center.x, pix_center.y),
            radius=30,
            fill=False,
        )
    )

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img_scaled)

    im1 = ax1.imshow(img_scaled, vmin=vmin, vmax=vmax)
    # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    for cx, cy in cs:
        line_color = next(ax1._get_lines.prop_cycler)["color"]
        ax1.axvline(cx, alpha=0.7, color=line_color)
        ax1.axhline(cy, alpha=0.9, color=line_color)
    # ax1.axvline(px, color="b", alpha=0.2)
    # ax1.axhline(py, color="b", alpha=0.2)

    plt.show()


def show_background_subtraction(
    before,
    after,
    bg_aperture_x,
    bg_aperture_y,
    bg_aperture_radius,
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    zscale = ZScaleInterval()
    vmin1, vmax1 = zscale.get_limits(before)

    im1 = ax1.imshow(before, vmin=vmin1, vmax=vmax1)
    im2 = ax2.imshow(after, vmin=vmin1, vmax=vmax1)

    fig.colorbar(im1)
    fig.colorbar(im2)

    image_center_row = int(np.floor(before.shape[0] / 2))
    image_center_col = int(np.floor(before.shape[1] / 2))
    print(f"Image center: {image_center_col}, {image_center_row}")
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
    # TODO: menu here for type of BG method
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


def determine_comet_center(uw1: SwiftUVOTImage, uvv: SwiftUVOTImage):
    # TODO: compare the results between uw1 and uvv and make sure they're the same
    # TODO: uvv tends to pick up the dust tail so we might expect some difference depending on the method of center detection,
    # so maybe we just use the uw1 and assume it's less likely to have a tail to scramble the center-finding
    print("Determining center of comet:")
    pix_center = get_uvot_image_center(img=uw1)
    search_ap = CircularAperture((pix_center.x, pix_center.y), r=30)
    pixel_center = find_comet_center(
        img=uw1,
        method=CometCenterFindingMethod.pixel_center,
        search_aperture=search_ap,
    )
    centroid = find_comet_center(
        img=uw1,
        method=CometCenterFindingMethod.aperture_centroid,
        search_aperture=search_ap,
    )
    peak = find_comet_center(
        img=uw1,
        method=CometCenterFindingMethod.aperture_peak,
        search_aperture=search_ap,
    )
    print("\tBy image center: ", pixel_center)
    print(
        "\tBy centroid (center of mass) in aperture radius 30 at image center: ",
        centroid,
    )
    print("\tBy peak value in aperture radius 30 at image center: ", peak)

    # show_centers(uw1, [pixel_center, centroid, peak])
    # TODO: return center_uw1, center_uvv


def q_vs_aperture_radius(epoch: Epoch, uw1: SwiftUVOTImage, uvv: SwiftUVOTImage):
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    aperture_radii, r_step = np.linspace(1, 60, num=300, retstep=True)
    rednesses = [10]
    dust_redness_list = list(
        # map(lambda x: DustReddeningPercent(x), [0, 5, 10, 15, 20, 25])
        map(lambda x: DustReddeningPercent(x), rednesses)
    )
    redness_to_beta = {x.reddening: beta_parameter(x) for x in dust_redness_list}

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
            "dust_redness": list(map(lambda x: int(x.reddening), reds)),
            "counts_uw1": count_uw1,
            "counts_uvv": count_uvv,
            "flux_OH": flux,
            "num_OH": oh,
            "Q_H2O": qs,
        }
    )

    qmask = df["Q_H2O"] > 0
    print(df[qmask])

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
                # print(positive_Qs)
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

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[2].set_yscale("log")
    df.plot.line(x="aperture_radius", y="counts_uw1", subplots=True, ax=axs[0])
    df.plot.line(x="aperture_radius", y="counts_uvv", subplots=True, ax=axs[1])
    df.plot.line(x="aperture_radius", y="Q_H2O", subplots=True, ax=axs[2])

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


def do_comet_photometry(
    img: SwiftUVOTImage,
    aperture_radius: float,
):
    pix_center = get_uvot_image_center(img=img)
    ap_x, ap_y = pix_center.x, pix_center.y
    comet_aperture = CircularAperture((ap_x, ap_y), r=aperture_radius)

    comet_count_rate = float(ApertureStats(img, comet_aperture).sum)

    return comet_count_rate


def extract_profile(img: SwiftUVOTImage, r: int, theta: float, plot_profile: bool):
    pix_center = get_uvot_image_center(img=img)
    search_aperture = CircularAperture((pix_center.x, pix_center.y), r=r)
    # comet_center = get_uvot_image_center(img)

    comet_center = find_comet_center(
        img=img,
        method=CometCenterFindingMethod.aperture_peak,
        search_aperture=search_aperture,
    )
    x0 = comet_center[0] - r * np.cos(theta)
    y0 = comet_center[1] - r * np.sin(theta)
    x1 = comet_center[0] + r * np.cos(theta)
    y1 = comet_center[1] + r * np.sin(theta)

    num_samples = 2 * r + 1

    xs = np.linspace(np.round(x0), np.round(x1), num_samples)
    ys = np.linspace(np.round(y0), np.round(y1), num_samples)

    profile = img[ys.astype(np.int32), xs.astype(np.int32)]

    distances_from_center = np.array(range(-r, r + 1))

    if plot_profile is True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(distances_from_center, profile)

        plt.show()
        plt.close()

    return xs, ys, profile


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

    bguw1 = get_background(uw1_sum)
    bguvv = get_background(uvv_sum)
    # bguw1 = get_background(uw1_median)
    # bguvv = get_background(uvv_median)
    print("")
    print(f"Background count rate for uw1: {bguw1}")
    print(f"Background count rate for uvv: {bguvv}")

    # uw1 = np.clip(uw1_sum - bguw1.count_rate_per_pixel, 0, None)
    # uvv = np.clip(uvv_sum - bguvv.count_rate_per_pixel, 0, None)
    uw1 = uw1_sum - bguw1.count_rate_per_pixel
    uvv = uvv_sum - bguvv.count_rate_per_pixel
    # uw1 = uw1_median - bguw1.count_rate_per_pixel
    # uvv = uvv_median - bguvv.count_rate_per_pixel

    determine_comet_center(uw1, uvv)

    df = q_vs_aperture_radius(epoch=epoch, uw1=uw1, uvv=uvv)

    # show_fits_subtracted(uw1=uw1_sum, uvv=uvv_sum, beta=beta)

    xs_list = []
    ys_list = []
    zs_list = []

    profile_radius = 30
    # angles = [3 * np.pi / 2, np.pi / 2, np.pi / 4]
    angles = np.linspace(0, np.pi, 50)
    angles = angles[:-1]
    for theta in angles:
        xs, ys, prof = extract_profile(
            img=uvv, r=profile_radius, theta=theta, plot_profile=False
        )
        xs_list.extend(xs)
        ys_list.extend(ys)
        zs_list.extend(prof)

    pix_center = get_uvot_image_center(img=uvv)
    search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    comet_center = find_comet_center(
        img=uvv,
        method=CometCenterFindingMethod.aperture_peak,
        search_aperture=search_aperture,
    )

    mx, my = np.meshgrid(
        np.linspace(
            comet_center[0] - profile_radius / 2, comet_center[0] + profile_radius / 2
        ),
        np.linspace(
            comet_center[1] - profile_radius / 2, comet_center[1] + profile_radius / 2
        ),
    )
    mz = interp.griddata((xs_list, ys_list), zs_list, (mx, my), method="linear")

    ax1.plot_surface(mx, my, mz, cmap="magma")

    # ax.plot_trisurf(xs_list, ys_list, zs_list, cmap="magma")
    ax2.contour(mx, my, mz, cmap="magma")

    plt.show()


if __name__ == "__main__":
    sys.exit(main())
