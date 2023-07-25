#!/usr/bin/env python3

import os
import glob
import pathlib
import sys
import itertools
import copy
import logging as log
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)

from argparse import ArgumentParser

from swift_types import (
    SwiftData,
    SwiftFilter,
    SwiftOrbitID,
    SwiftObservationLog,
    SwiftPixelResolution,
    SwiftUVOTImage,
    StackingMethod,
    filter_to_string,
)
from configs import read_swift_project_config
from stacking import (
    stack_image_by_selection,
    write_stacked_image,
    includes_uvv_and_uw1_filters,
)
from observation_log import read_observation_log
from stack_info import stackinfo_from_stacked_images
from typing import Tuple, List, Optional

from user_input import get_selection


__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
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


def image_mid_row_col(img: SwiftUVOTImage) -> Tuple[int, int]:
    mid_row = int(np.floor(img.shape[0] / 2))
    mid_col = int(np.floor(img.shape[1] / 2))

    return (mid_row, mid_col)


def show_fits_scaled(image_list: List[SwiftUVOTImage], stacking_method: StackingMethod):
    num_images = len(image_list)
    fig, axes = plt.subplots(1, num_images, figsize=(100, 20))

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_list[0])

    for ax, img in zip(axes, image_list):
        im = ax.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        fig.colorbar(im)
        ax.axvline(int(np.floor(img.shape[1] / 2)), color="b", alpha=0.1)
        ax.axhline(int(np.floor(img.shape[0] / 2)), color="b", alpha=0.1)

    plt.savefig(f"padding_before_after_{str(stacking_method)}.png")
    plt.show()
    plt.close()


def pad_to_match_sizes(
    uw1: SwiftUVOTImage, uvv: SwiftUVOTImage, stacking_method: StackingMethod
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
    #     f"For {stacking_method}:\tuw1 transformed from {uw1copy.shape} ==> {uw1.shape}\t\tuvv transformed from {uvvcopy.shape} ==> {uvv.shape}"
    # )

    uw1_mid_row_original, uw1_mid_col_original = image_mid_row_col(uw1copy)
    uw1_center_pixel_original = uw1copy[uw1_mid_row_original, uw1_mid_col_original]
    # print(
    #     f"Original uw1 center:\t{uw1_mid_row_original} {uw1_mid_col_original}\t\tPixel value at center: {uw1_center_pixel_original}"
    # )
    uw1_mid_row, uw1_mid_col = image_mid_row_col(uw1)
    # uw1_center_pixel = uw1[uw1_mid_row, uw1_mid_col]
    # print(
    #     f"Padded uw1 center:\t{uw1_mid_row} {uw1_mid_col}\t\tPixel value at center: {uw1_center_pixel}"
    # )

    uvv_mid_row_original, uvv_mid_col_original = image_mid_row_col(uvvcopy)
    uvv_center_pixel_original = uvvcopy[uvv_mid_row_original, uvv_mid_col_original]
    # print(
    #     f"Original uvv center:\t{uvv_mid_row_original} {uvv_mid_col_original}\t\tPixel value at center: {uvv_center_pixel_original}"
    # )
    uvv_mid_row, uvv_mid_col = image_mid_row_col(uvv)
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

    # show_fits_scaled([uw1, uw1copy, uvv, uvvcopy], stacking_method=stacking_method)

    return (uw1, uvv)


def do_stack(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    stacked_image_dir: pathlib.Path,
    stackinfo_output_path: pathlib.Path,
    do_coincidence_correction: bool,
    detector_scale: SwiftPixelResolution,
) -> None:
    # create the directory if it doesn't exist
    stacked_image_dir.mkdir(parents=True, exist_ok=True)

    # test if there are uvv and uw1 images in the data set
    (stackable, _) = includes_uvv_and_uw1_filters(obs_log=obs_log)
    if not stackable:
        print("The selection does not have data in both uw1 and uvv filters!")

    # Do both filters with sum and median stacking
    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]

    stacking_outputs = {}
    stacked_images = {}

    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        print(
            f"Stacking for filter {filter_to_string(filter_type)}: stacking type {stacking_method} ..."
        )

        # now narrow down the data to just one filter at a time
        filter_mask = obs_log["FILTER"] == filter_type
        ml = obs_log[filter_mask]

        stacked_images[(filter_type, stacking_method)] = stack_image_by_selection(
            swift_data=swift_data,
            obs_log=ml,
            do_coincidence_correction=do_coincidence_correction,
            detector_scale=detector_scale,
            stacking_method=stacking_method,
        )

        if stacked_images[(filter_type, stacking_method)] is None:
            print("Stacking image failed, skipping... ")
            continue

    # Adjust the images from each filter to be the same size
    print("Padding uw1 and uvv images to match dimensions ...")
    for stacking_method in stacking_methods:
        (uw1_img, uvv_img) = pad_to_match_sizes(
            uw1=stacked_images[(SwiftFilter.uw1, stacking_method)].stacked_image,
            uvv=stacked_images[(SwiftFilter.uvv, stacking_method)].stacked_image,
            stacking_method=stacking_method,
        )
        stacked_images[(SwiftFilter.uw1, stacking_method)].stacked_image = uw1_img
        stacked_images[(SwiftFilter.uvv, stacking_method)].stacked_image = uvv_img

    # print("Writing stack results ...")
    # Write the padded images and stacking information
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        stacked_path, stacked_info_path = write_stacked_image(
            stacked_image_dir=stacked_image_dir,
            stacked_image=stacked_images[(filter_type, stacking_method)],
        )
        # print(f"Wrote to {stacked_path}, info at {stacked_info_path}")

        stacking_outputs[(filter_type, stacking_method)] = (
            str(stacked_path.name),
            str(stacked_info_path.name),
        )

    stackinfo_from_stacked_images(stackinfo_output_path, stacking_outputs)


def select_epoch_to_stack(epoch_dir: pathlib.Path) -> Tuple[SwiftObservationLog, str]:
    glob_pattern = str(epoch_dir / pathlib.Path("*.parquet"))

    epoch_filename_list = sorted(glob.glob(glob_pattern))

    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])
    # obs_log = pd.read_parquet(epoch_path)
    obs_log = read_observation_log(epoch_path)
    return obs_log, epoch_path.stem


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_project_config.swift_data_path)
        .expanduser()
        .resolve()
    )

    epoch_to_stack, stack_basename = select_epoch_to_stack(
        swift_project_config.product_save_path.expanduser().resolve()
        / pathlib.Path("epochs")
    )

    stacked_image_dir = (
        swift_project_config.product_save_path.expanduser().resolve()
        / pathlib.Path("stacked")
        / pathlib.Path(stack_basename)
    )

    stackinfo_output_path = stacked_image_dir / pathlib.Path("stack.json")

    print(stackinfo_output_path)

    do_stack(
        swift_data=swift_data,
        obs_log=epoch_to_stack,
        stacked_image_dir=stacked_image_dir,
        stackinfo_output_path=stackinfo_output_path,
        do_coincidence_correction=True,
        detector_scale=SwiftPixelResolution.event_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
