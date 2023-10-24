#!/usr/bin/env python3

from collections import UserDict
import copy
import pathlib
from typing import Tuple, Optional, List, TypeAlias
from enum import StrEnum

import numpy as np
import logging as log

from astropy.io import fits
from tqdm import tqdm
from dataclasses import dataclass

from swift_data import SwiftData
from swift_filter import SwiftFilter
from uvot_image import SwiftUVOTImage, PixelCoord, SwiftPixelResolution
from epochs import Epoch
from coincidence_correction import coincidence_correction


__all__ = [
    "StackingMethod",
    "get_image_dimensions_to_center_comet",
    "determine_stacking_image_size",
    "center_image_on_coords",
    "stack_epoch",
]


class StackingMethod(StrEnum):
    summation = "sum"
    median = "median"

    @classmethod
    def all_stacking_methods(cls):
        return [x for x in cls]


StackedUVOTImageSet: TypeAlias = dict[
    tuple[SwiftFilter, StackingMethod], SwiftUVOTImage
]


def get_image_dimensions_to_center_comet(
    source_image: SwiftUVOTImage, source_coords_to_center: PixelCoord
) -> Tuple[float, float]:
    """
    If we want to re-center the source_image on source_coords_to_center, we need to figure out the dimensions the new image
    needs to be to fit the old picture after we shift it over to its new position
    Returns tuple of (rows, columns) the new image would have to be
    """
    num_rows, num_columns = source_image.shape
    center_row = num_rows / 2.0
    center_col = num_columns / 2.0

    # distance the image needs to move
    d_rows = center_row - source_coords_to_center.y
    d_cols = center_col - source_coords_to_center.x

    # the total padding is twice the distance it needs to move -
    # extra space for it to move into, and the empty space it would leave behind
    row_padding = 2 * d_rows
    col_padding = 2 * d_cols

    new_rows_cols = (
        np.ceil(source_image.shape[0] + np.abs(row_padding)),
        np.ceil(source_image.shape[1] + np.abs(col_padding)),
    )

    return new_rows_cols


def determine_stacking_image_size(
    swift_data: SwiftData,
    epoch: Epoch,
) -> Optional[Tuple[int, int]]:
    """Opens every FITS file specified in the given observation log and finds the image size necessary for stacking all of the images"""

    # how far in pixels each comet image needs to shift
    image_dimensions_list = []

    for _, row in epoch.iterrows():
        image_dir = swift_data.get_uvot_image_directory(row["OBS_ID"])  # type: ignore

        # open file
        img_file = image_dir / pathlib.Path(str(row["FITS_FILENAME"]))
        image_data = fits.getdata(img_file, ext=row["EXTENSION"])

        # keep a list of the image sizes
        image_dimensions = get_image_dimensions_to_center_comet(
            image_data, PixelCoord(x=row["PX"], y=row["PY"])  # type: ignore
        )
        image_dimensions_list.append(image_dimensions)

    if len(image_dimensions_list) == 0:
        print("No images found in epoch!")
        return None

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(image_dimensions_list, key=lambda k: k[0], reverse=True)[0][0]
    max_num_cols = sorted(image_dimensions_list, key=lambda k: k[1], reverse=True)[0][1]

    # how many extra pixels we need
    return (max_num_rows, max_num_cols)


def center_image_on_coords(
    source_image: SwiftUVOTImage,
    source_coords_to_center: PixelCoord,
    stacking_image_size: Tuple[int, int],
) -> SwiftUVOTImage:
    """
    size is the (rows, columns) size of the positive quandrant of the new image
    """

    center_x, center_y = np.round(source_coords_to_center.x), np.round(
        source_coords_to_center.y
    )
    new_r, new_c = map(lambda x: int(x), np.ceil(stacking_image_size))

    # enforce that we have an odd number of pixels so that the comet can be moved to the center row, center column
    if new_r % 2 == 0:
        half_r = new_r / 2
        new_r = new_r + 1
    else:
        half_r = (new_r - 1) / 2
    if new_c % 2 == 0:
        half_c = new_c / 2
        new_c = new_c + 1
    else:
        half_c = (new_c - 1) / 2

    # create empty array to hold the new, centered image
    centered_image = np.zeros((new_r, new_c))

    def shift_row(r):
        return int(r + half_r + 0 - center_y)

    def shift_column(c):
        return int(c + half_c + 0 - center_x)

    for r in range(source_image.shape[0]):
        for c in range(source_image.shape[1]):
            centered_image[shift_row(r), shift_column(c)] = source_image[r, c]

    log.debug(">> center_image_on_coords: image center at (%s, %s)", half_c, half_r)
    log.debug(
        ">> center_image_on_coords: Shifted comet center to coordinates (%s, %s): this should match image center",
        shift_column(center_x),
        shift_row(center_y),
    )

    return centered_image


# TODO: this could just return the sum and median images together
def stack_epoch(
    swift_data: SwiftData,
    epoch: Epoch,
    stacking_method: StackingMethod = StackingMethod.summation,
    do_coincidence_correction: bool = True,
    pixel_resolution: SwiftPixelResolution = SwiftPixelResolution.data_mode,
) -> Optional[SwiftUVOTImage]:
    """
    Blindly takes every entry in the given Epoch and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    """

    # determine how big our stacked image needs to be
    stacking_image_size = determine_stacking_image_size(
        swift_data=swift_data,
        epoch=epoch,
    )

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    image_data_to_stack: List[SwiftUVOTImage] = []
    exposure_times: List[float] = []

    stacking_progress_bar = tqdm(epoch.iterrows(), total=len(epoch), unit="images")
    for _, row in stacking_progress_bar:
        obsid = row["OBS_ID"]

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore

        exp_time = float(row["EXPOSURE"])

        # read the image
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        # center the comet
        image_data = center_image_on_coords(
            source_image=image_data,  # type: ignore
            source_coords_to_center=PixelCoord(x=row["PX"], y=row["PY"]),  # type: ignore
            stacking_image_size=stacking_image_size,  # type: ignore
        )

        # do any processing before stacking
        if do_coincidence_correction:
            # the correction expects images in count rate, but we are storing the raw images so divide by exposure time here
            coi_map = coincidence_correction(
                img_data=image_data / exp_time, scale=pixel_resolution
            )
            image_data = image_data * coi_map

        image_data_to_stack.append(image_data)
        exposure_times.append(exp_time)

        stacking_progress_bar.set_description(
            f"{image_path.name} extension {row.EXTENSION}"
        )

    exposure_time = epoch.EXPOSURE.sum()

    if stacking_method == StackingMethod.summation:
        stacked_image = np.sum(image_data_to_stack, axis=0) / exposure_time
    elif stacking_method == StackingMethod.median:
        imgs_copy = copy.deepcopy(image_data_to_stack)
        for img, exp_time in zip(imgs_copy, exposure_times):
            img /= exp_time
        stacked_image = np.median(imgs_copy, axis=0)
    else:
        log.info("Invalid stacking method specified, defaulting to summation...")
        return None

    return stacked_image
