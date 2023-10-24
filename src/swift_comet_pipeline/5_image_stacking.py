#!/usr/bin/env python3

import os
import pathlib
import sys
import itertools
import copy
import logging as log
import numpy as np
from argparse import ArgumentParser
from typing import Tuple
from itertools import product

import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.configs import read_swift_project_config
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
    filter_to_string,
)
from swift_comet_pipeline.stacking import (
    StackedUVOTImageSet,
    StackingMethod,
    stack_epoch,
)
from swift_comet_pipeline.epochs import Epoch
from swift_comet_pipeline.tui import get_yes_no, epoch_menu, bool_to_x_or_check
from swift_comet_pipeline.uvot_image import (
    SwiftUVOTImage,
    get_uvot_image_center,
    get_uvot_image_center_row_col,
)


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


def show_fits_scaled(stacked_image_set: StackedUVOTImageSet):
    _, axes = plt.subplots(2, 2, figsize=(100, 20))

    zscale = ZScaleInterval()

    filter_types = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    for i, (filter_type, stacking_method) in enumerate(
        product(filter_types, stacking_methods)
    ):
        img = stacked_image_set[(filter_type, stacking_method)]
        vmin, vmax = zscale.get_limits(img)
        ax = axes[np.unravel_index(i, axes.shape)]
        ax.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        # fig.colorbar(im)
        # TODO: get center of image from function in uvot
        ax.axvline(int(np.floor(img.shape[1] / 2)), color="b", alpha=0.1)
        ax.axhline(int(np.floor(img.shape[0] / 2)), color="b", alpha=0.1)
        ax.set_title(f"{filter_to_file_string(filter_type)} {stacking_method}")

    plt.show()
    plt.close()


def pad_to_match_sizes(
    uw1: SwiftUVOTImage, uvv: SwiftUVOTImage
) -> Tuple[SwiftUVOTImage, SwiftUVOTImage]:
    """
    Given two images, pad the smaller image so that the uw1 and uvv images end up the same size
    """
    uw1copy = copy.deepcopy(uw1)
    uvvcopy = copy.deepcopy(uvv)

    cols_to_add = round((uw1.shape[1] - uvv.shape[1]) / 2)
    rows_to_add = round((uw1.shape[0] - uvv.shape[0]) / 2)

    if cols_to_add > 0:
        # uw1 is larger, pad uvv to be larger
        uvv = np.pad(
            uvv,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        # uvv is larger, pad uw1 to be larger
        cols_to_add = np.abs(cols_to_add)
        uw1 = np.pad(
            uw1,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )

    if rows_to_add > 0:
        # uw1 is larger, pad uvv to be larger
        uvv = np.pad(
            uvv,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        # uvv is larger, pad uw1 to be larger
        rows_to_add = np.abs(rows_to_add)
        uw1 = np.pad(
            uw1,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    # print(
    #     f"uw1 transformed from {uw1copy.shape} ==> {uw1.shape}\t\tuvv transformed from {uvvcopy.shape} ==> {uvv.shape}"
    # )

    uw1_mid_row_original, uw1_mid_col_original = get_uvot_image_center_row_col(uw1copy)
    uw1_center_pixel_original = uw1copy[uw1_mid_row_original, uw1_mid_col_original]
    # print(
    #     f"Original uw1 center:\t{uw1_mid_row_original} {uw1_mid_col_original}\t\tPixel value at center: {uw1_center_pixel_original}"
    # )
    uw1_mid_row, uw1_mid_col = get_uvot_image_center_row_col(uw1)
    # uw1_center_pixel = uw1[uw1_mid_row, uw1_mid_col]
    # print(
    #     f"Padded uw1 center:\t{uw1_mid_row} {uw1_mid_col}\t\tPixel value at center: {uw1_center_pixel}"
    # )

    uvv_mid_row_original, uvv_mid_col_original = get_uvot_image_center_row_col(uvvcopy)
    uvv_center_pixel_original = uvvcopy[uvv_mid_row_original, uvv_mid_col_original]
    # print(
    #     f"Original uvv center:\t{uvv_mid_row_original} {uvv_mid_col_original}\t\tPixel value at center: {uvv_center_pixel_original}"
    # )
    uvv_mid_row, uvv_mid_col = get_uvot_image_center_row_col(uvv)
    # uvv_center_pixel = uvv[uvv_mid_row, uvv_mid_col]
    # print(
    #     f"Padded uvv center:\t{uvv_mid_row} {uvv_mid_col}\t\tPixel value at center: {uvv_center_pixel}"
    # )

    pixmatch_list_uw1 = list(zip(*np.where(uw1 == uw1_center_pixel_original)))
    # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
    if (uw1_mid_row, uw1_mid_col) not in pixmatch_list_uw1:
        print("Error padding uw1 image! This is a bug!")
        print(
            f"Pixel coordinates of new uw1 image that match center of old uw1 image: {pixmatch_list_uw1}"
        )
    # else:
    #     print("No errors padding uw1 image - center pixels match values")

    pixmatch_list_uvv = list(zip(*np.where(uvv == uvv_center_pixel_original)))
    # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
    if (uvv_mid_row, uvv_mid_col) not in pixmatch_list_uvv:
        print("Error padding uvv image! This is a bug!")
        print(
            f"Pixel coordinates of new uvv image that match center of old uvv image: {pixmatch_list_uvv}"
        )
    # else:
    #     print("No errors padding uvv image - center pixels match values")

    return (uw1, uvv)


def epoch_stacked_image_to_fits(epoch: Epoch, img: SwiftUVOTImage) -> fits.ImageHDU:
    hdu = fits.ImageHDU(data=img)

    # TODO: include data mode or event mode here

    hdr = hdu.header
    hdr["distunit"] = "AU"
    hdr["v_unit"] = "km/s"
    hdr["delta"] = np.mean(epoch.OBS_DIS)
    hdr["rh"] = np.mean(epoch.HELIO)
    hdr["ra_obj"] = np.mean(epoch.RA_OBJ)
    hdr["dec_obj"] = np.mean(epoch.DEC_OBJ)

    pix_center = get_uvot_image_center(img=img)
    hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y
    hdr["phase"] = np.mean(epoch.PHASE)

    dt = Time(np.max(epoch.MID_TIME)) - Time(np.min(epoch.MID_TIME))
    first_obs_row = epoch.loc[epoch.MID_TIME.idxmin()]
    last_obs_row = epoch.loc[epoch.MID_TIME.idxmax()]

    first_obs_time = Time(first_obs_row.MID_TIME)
    first_obs_time.format = "fits"
    hdr["firstobs"] = first_obs_time.value
    last_obs_time = Time(last_obs_row.MID_TIME)
    last_obs_time.format = "fits"
    hdr["lastobs"] = last_obs_time.value
    mid_obs = Time(np.mean(epoch.MID_TIME))
    mid_obs.format = "fits"
    hdr["mid_obs"] = mid_obs.value

    rh_start = first_obs_row.HELIO * u.AU
    rh_end = last_obs_row.HELIO * u.AU
    dr_dt = (rh_end - rh_start) / dt

    ddelta_dt = (last_obs_row.OBS_DIS * u.AU - first_obs_row.OBS_DIS * u.AU) / dt

    hdr["drh_dt"] = dr_dt.to_value(u.km / u.s)
    hdr["ddeltadt"] = ddelta_dt.to_value(u.km / u.s)

    return hdu


def do_stack(
    pipeline_files: PipelineFiles,
    swift_data: SwiftData,
    epoch: Epoch,
    epoch_path: pathlib.Path,
    do_coincidence_correction: bool,
    ask_to_save_stack: bool,
    show_stacked_images: bool,
) -> None:
    # Do both filters with sum and median stacking
    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]

    if epoch.DATAMODE.nunique() != 1:
        print("Images in the requested stack have mixed data modes!  Exiting.")
        exit(1)

    epoch_pixel_resolution = epoch.DATAMODE[0]
    stacked_images = StackedUVOTImageSet({})

    # do the stacking
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        fits_product = pipeline_files.stacked_image_products[  # type: ignore
            epoch_path, filter_type, stacking_method
        ]
        if fits_product.product_path.exists():
            print(f"Stack {fits_product.product_path} exists! Skipping.")
            return
        print(
            f"Stacking for filter {filter_to_string(filter_type)}: stacking type {stacking_method} ..."
        )

        # now narrow down the data to just one filter at a time
        filter_mask = epoch["FILTER"] == filter_type
        ml = epoch[filter_mask]

        stacked_img = stack_epoch(
            swift_data=swift_data,
            epoch=ml,
            stacking_method=stacking_method,
            do_coincidence_correction=do_coincidence_correction,
            pixel_resolution=epoch_pixel_resolution,
        )
        if stacked_img is None:
            print("Stacking image failed, skipping... ")
            return
        else:
            stacked_images[(filter_type, stacking_method)] = stacked_img

    # Adjust the images from each filter to be the same size
    for stacking_method in stacking_methods:
        (uw1_img, uvv_img) = pad_to_match_sizes(
            uw1=stacked_images[(SwiftFilter.uw1, stacking_method)],
            uvv=stacked_images[(SwiftFilter.uvv, stacking_method)],
        )
        stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
        stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

    if show_stacked_images:
        # show them
        show_fits_scaled(stacked_images)

    if ask_to_save_stack:
        print("Save results?")
        save_results = get_yes_no()
        if not save_results:
            return

    # get the stacked epoch pipeline product, stuff the epoch in it, and save
    stacked_epoch_product = pipeline_files.stacked_epoch_products[epoch_path]  # type: ignore
    stacked_epoch_product.data_product = epoch
    stacked_epoch_product.save_product()

    # write the stacked images as .FITS files
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        fits_product = pipeline_files.stacked_image_products[  # type: ignore
            epoch_path, filter_type, stacking_method
        ]
        print(f"Writing to {fits_product.product_path} ...")
        hdu = epoch_stacked_image_to_fits(
            epoch=epoch, img=stacked_images[(filter_type, stacking_method)]
        )
        fits_product.data_product = hdu
        fits_product.save_product()


def print_stacked_images_summary(
    pipeline_files: PipelineFiles,
) -> dict[pathlib.Path, bool]:
    is_epoch_stacked: dict[pathlib.Path, bool] = {}
    print("Summary of detected stacked images:")
    # loop through each epoch and look for associated stacked files
    for x in pipeline_files.epoch_file_paths:  # type: ignore
        ep_prod = pipeline_files.stacked_epoch_products[x]  # type: ignore
        print(ep_prod.product_path.stem, "\t", bool_to_x_or_check(ep_prod.exists()))
        is_epoch_stacked[x] = ep_prod.exists()

    return is_epoch_stacked


def menu_stack_all_or_selection() -> str:
    user_selection = None
    default_selection = "s"

    print("Stack all or make a selection? (a/s)")
    print("Default: selection")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or "s":
            user_selection = raw_selection
        if raw_selection == "":
            user_selection = default_selection

    return user_selection


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    is_epoch_stacked = print_stacked_images_summary(pipeline_files=pipeline_files)
    if all(is_epoch_stacked.values()):
        print("Everything stacked! Nothing to do.")
        return 0

    menu_selection = menu_stack_all_or_selection()

    epochs_to_stack = []
    ask_to_save_stack = True
    show_stacked_images = True
    if menu_selection == "a":
        epochs_to_stack = pipeline_files.epoch_products
        ask_to_save_stack = False
        show_stacked_images = False
    elif menu_selection == "s":
        epochs_to_stack = [epoch_menu(pipeline_files)]

    if epochs_to_stack is None:
        print("Pipeline error! This is a bug with pipeline_files.epoch_products!")
        return 1

    for epoch_product in epochs_to_stack:
        if epoch_product is None:
            print("Error selecting epoch! Exiting.")
            return 1
        epoch_product.load_product()
        epoch = epoch_product.data_product

        non_vetoed_epoch = epoch[epoch.manual_veto == np.False_]

        do_stack(
            pipeline_files=pipeline_files,
            swift_data=swift_data,
            epoch=non_vetoed_epoch,
            epoch_path=epoch_product.product_path,
            do_coincidence_correction=True,
            ask_to_save_stack=ask_to_save_stack,
            show_stacked_images=show_stacked_images,
        )


if __name__ == "__main__":
    sys.exit(main())
