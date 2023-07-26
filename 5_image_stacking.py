#!/usr/bin/env python3

import os
import glob
import pathlib
import sys
import itertools
import copy
import logging as log
import numpy as np
import pyarrow as pa
from argparse import ArgumentParser
from typing import Tuple, List

import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits

from swift_types import (
    SwiftData,
    SwiftFilter,
    SwiftPixelResolution,
    SwiftUVOTImage,
    StackingMethod,
    filter_to_file_string,
    filter_to_string,
)
from configs import read_swift_project_config
from stacking import stack_epoch
from epochs import Epoch, read_epoch, write_epoch
from user_input import get_selection, get_yes_no


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


def includes_uvv_and_uw1_filters(
    epoch: Epoch,
) -> bool:
    """
    To find OH and perform dust subtraction we need data from the UV and UW1 filter from somewhere across the given data set in orbit_ids.
    Returns a list of orbits that have UV or UW1 images, after removing orbits that have no data in the UV or UW1 filters
    """

    has_uvv_filter = epoch[epoch["FILTER"] == SwiftFilter.uvv]
    has_uvv_set = set(has_uvv_filter["ORBIT_ID"])

    has_uw1_filter = epoch[epoch["FILTER"] == SwiftFilter.uw1]
    has_uw1_set = set(has_uw1_filter["ORBIT_ID"])

    has_both = len(has_uvv_set) > 0 and len(has_uw1_set) > 0

    # contributing_orbits = has_uvv_set
    # contributing_orbits.update(has_uw1_set)

    # return (has_both, list(contributing_orbits))
    return has_both


def image_mid_row_col(img: SwiftUVOTImage) -> Tuple[int, int]:
    mid_row = int(np.floor(img.shape[0] / 2))
    mid_col = int(np.floor(img.shape[1] / 2))

    return (mid_row, mid_col)


def show_fits_scaled(image_list: List[SwiftUVOTImage]):
    # num_images = len(image_list)
    _, axes = plt.subplots(2, 2, figsize=(100, 20))

    zscale = ZScaleInterval()

    for ax, img in zip(axes.reshape(-1), image_list):
        vmin, vmax = zscale.get_limits(img)
        ax.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        # fig.colorbar(im)
        ax.axvline(int(np.floor(img.shape[1] / 2)), color="b", alpha=0.1)
        ax.axhline(int(np.floor(img.shape[0] / 2)), color="b", alpha=0.1)

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

    return (uw1, uvv)


def do_stack(
    swift_data: SwiftData,
    epoch: Epoch,
    epoch_name: str,
    stack_dir_path: pathlib.Path,
    do_coincidence_correction: bool,
    detector_scale: SwiftPixelResolution,
) -> None:
    # test if there are uvv and uw1 images in the data set
    if not includes_uvv_and_uw1_filters(epoch=epoch):
        print("The selection does not have data in both uw1 and uvv filters!")

    # Do both filters with sum and median stacking
    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]

    stacked_images = {}

    # do the stacking
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        print(
            f"Stacking for filter {filter_to_string(filter_type)}: stacking type {stacking_method} ..."
        )
        fits_file_path = stack_dir_path / pathlib.Path(
            f"{epoch_name}_{filter_to_file_string(filter_type)}_{stacking_method}.fits"
        )
        if fits_file_path.exists():
            print(f"Stack {fits_file_path} exists! Exiting.")
            return

        # now narrow down the data to just one filter at a time
        filter_mask = epoch["FILTER"] == filter_type
        ml = epoch[filter_mask]

        stacked_images[(filter_type, stacking_method)] = stack_epoch(
            swift_data=swift_data,
            epoch=ml,
            stacking_method=stacking_method,
            do_coincidence_correction=do_coincidence_correction,
            detector_scale=detector_scale,
        )

        if stacked_images[(filter_type, stacking_method)] is None:
            print("Stacking image failed, skipping... ")
            continue

    # Adjust the images from each filter to be the same size
    for stacking_method in stacking_methods:
        (uw1_img, uvv_img) = pad_to_match_sizes(
            uw1=stacked_images[(SwiftFilter.uw1, stacking_method)],
            uvv=stacked_images[(SwiftFilter.uvv, stacking_method)],
        )
        stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
        stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

    # show them
    show_fits_scaled(
        [
            stacked_images[(SwiftFilter.uw1, StackingMethod.summation)],
            stacked_images[(SwiftFilter.uw1, StackingMethod.median)],
            stacked_images[(SwiftFilter.uvv, StackingMethod.summation)],
            stacked_images[(SwiftFilter.uvv, StackingMethod.median)],
        ]
    )

    print("Save results?")
    save_results = get_yes_no()
    if not save_results:
        return

    # attach metadata to our epoch about stacking parameters
    epoch_stack_schema = pa.schema(
        [],
        metadata={
            "coincidence_correction": str(do_coincidence_correction),
            "detector_scale": str(detector_scale),
        },
    )
    epoch_write_path = stack_dir_path / pathlib.Path(epoch_name + ".parquet")

    # write the filtered epoch as a record of contributing data
    write_epoch(
        epoch=epoch, epoch_path=epoch_write_path, additional_schema=epoch_stack_schema
    )

    # write the stacked images as .FITS files
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        fits_filename = (
            f"{epoch_name}_{filter_to_file_string(filter_type)}_{stacking_method}.fits"
        )
        fits_path = stack_dir_path / pathlib.Path(fits_filename)
        print(fits_path)
        hdu = fits.PrimaryHDU(stacked_images[(filter_type, stacking_method)])
        hdu.writeto(fits_path, overwrite=True)


def select_epoch(epoch_dir: pathlib.Path) -> pathlib.Path:
    glob_pattern = str(epoch_dir / pathlib.Path("*.parquet"))
    epoch_filename_list = sorted(glob.glob(glob_pattern))
    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])

    return epoch_path


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_dir_path = swift_project_config.epoch_dir_path
    if epoch_dir_path is None:
        print(f"Could not find epoch_path in {swift_project_config_path}, exiting.")
        return

    epoch_path = select_epoch(epoch_dir_path)
    epoch = read_epoch(epoch_path)

    stack_dir_path = swift_project_config.product_save_path / pathlib.Path("stacked")
    stack_dir_path.mkdir(parents=True, exist_ok=True)

    non_veto_epoch = epoch[epoch.manual_veto == np.False_]

    do_stack(
        swift_data=swift_data,
        epoch=non_veto_epoch,
        epoch_name=epoch_path.stem,
        stack_dir_path=stack_dir_path,
        do_coincidence_correction=True,
        detector_scale=SwiftPixelResolution.event_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
