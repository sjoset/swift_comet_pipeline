#!/usr/bin/env python3

import pathlib
import json
import numpy as np
import logging as log

from astropy.io import fits
from typing import Tuple, Optional, List
from dataclasses import asdict

from swift_types import (
    SwiftObservationID,
    SwiftObservationLog,
    SwiftFilter,
    SwiftUVOTImage,
    SwiftStackedUVOTImage,
    PixelCoord,
    filter_to_file_string,
    SwiftPixelResolution,
)
from swift_data import SwiftData
from swift_observation_log import (
    match_by_obsid_and_filter,
)
from coincidence_correction import coincidence_correction


__all__ = [
    "get_image_dimensions_to_center_comet",
    "determine_stacking_image_size",
    "center_image_on_coords",
    "coincidence_loss_correction",
    "stack_by_obsid",
]


__version__ = "0.0.1"


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

    retval = (
        np.ceil(source_image.shape[0] + np.abs(row_padding)),
        np.ceil(source_image.shape[1] + np.abs(col_padding)),
    )

    return retval


def determine_stacking_image_size(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    obsid: SwiftObservationID,
    filter_type: SwiftFilter,
) -> Optional[Tuple[int, int]]:
    image_dir = swift_data.get_uvot_image_directory(obsid)

    # dataframe holding the observation log entries for this obsid and this filter
    matching_observations = match_by_obsid_and_filter(obs_log, obsid, filter_type)

    if len(matching_observations) == 0:
        return None

    # how far in pixels each comet image needs to shift
    image_dimensions = []

    # loop through each dataframe row
    for _, row in matching_observations.iterrows():
        # open file
        img_file = image_dir / pathlib.Path(str(row["FITS_FILENAME"]))
        image_data = fits.getdata(img_file, ext=row["EXTENSION"])

        # keep a list of the image sizes
        dimensions = get_image_dimensions_to_center_comet(
            image_data, PixelCoord(x=row["PX"], y=row["PY"])  # type: ignore
        )
        image_dimensions.append(dimensions)

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(image_dimensions, key=lambda k: k[0], reverse=True)[0][0]
    max_num_cols = sorted(image_dimensions, key=lambda k: k[1], reverse=True)[0][1]

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


# TODO: move this over
def coincidence_loss_correction(image_data: SwiftUVOTImage) -> SwiftUVOTImage:
    return image_data


def stack_by_obsid(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    obsid: SwiftObservationID,
    filter_type: SwiftFilter,
    do_coincidence_correction: bool = True,
    detector_scale: SwiftPixelResolution = SwiftPixelResolution.data_mode,
) -> Optional[SwiftStackedUVOTImage]:
    """Returns the stacked image and the total exposure time as a tuple"""
    # get the observations we care about
    matching_obs_log = match_by_obsid_and_filter(
        obs_log=obs_log, obsid=obsid, filter_type=filter_type
    )

    # determine how big our stacked image needs to be
    stacking_image_size = determine_stacking_image_size(
        swift_data=swift_data,
        obs_log=matching_obs_log,
        obsid=obsid,
        filter_type=filter_type,
    )

    for _, row in matching_obs_log.iterrows():
        log.info(
            "Found extension %s of %s to stack ...",
            row["EXTENSION"],
            row["FITS_FILENAME"],
        )

    image_data_to_stack: List[SwiftUVOTImage] = []
    exposure_times: List[float] = []
    source_filenames: List[pathlib.Path] = []
    source_extensions: List[int] = []

    for _, row in matching_obs_log.iterrows():
        # Get list of image files for this filter and image type
        img_file_list = swift_data.get_swift_uvot_image_paths(
            obsid=obsid, filter_type=filter_type
        )
        if img_file_list is None:
            log.info("No files found for %s and %s", obsid, filter_type)
            continue

        for img_file in img_file_list:
            log.info(
                "Processing %s, extension %s: Comet center at %s, %s",
                img_file.name,
                row["EXTENSION"],
                row["PX"],
                row["PY"],
            )

            source_filenames.append(pathlib.Path(img_file.name))
            source_extensions.append(int(row["EXTENSION"]))

            # read the image
            image_data = fits.getdata(img_file, ext=row["EXTENSION"])

            # center the comet
            image_data = center_image_on_coords(
                source_image=image_data,  # type: ignore
                source_coords_to_center=PixelCoord(x=row["PX"], y=row["PY"]),  # type: ignore
                stacking_image_size=stacking_image_size,  # type: ignore
            )

            # do any processing before stacking
            if do_coincidence_correction:
                image_data = coincidence_correction(image_data, detector_scale)

            image_data_to_stack.append(image_data)
            exposure_times.append(float(row["EXPOSURE"]))

    # did we find any images?  Not guaranteed that we do
    if len(exposure_times) == 0:
        return None

    exposure_time = np.sum(exposure_times)
    stacked_image = np.sum(image_data_to_stack, axis=0)

    return SwiftStackedUVOTImage(
        stacked_image=stacked_image,
        sources=list(
            zip(
                # duplicate the obsid into a list of the appropriate length
                [obsid for _ in range(len(source_filenames))],
                source_filenames,
                source_extensions,
            )
        ),
        exposure_time=exposure_time,
        filter_type=filter_type,
    )


def stack_by_orbit(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    obsid: SwiftObservationID,
    filter_type: SwiftFilter,
    do_coincidence_correction: bool = True,
    detector_scale: SwiftPixelResolution = SwiftPixelResolution.data_mode,
) -> SwiftStackedUVOTImage:
    # matching_data =
    pass


def write_stacked_image(
    stacked_image_dir: pathlib.Path, image: SwiftStackedUVOTImage
) -> None:
    filter_str = filter_to_file_string(image.filter_type)

    contributing_obsids = np.unique([x[0] for x in image.sources])
    obsids_str = "_".join(contributing_obsids)

    # name the file after all of the contributing observation ids
    base_name = obsids_str + "_" + filter_str

    stacked_image_path = stacked_image_dir / pathlib.Path(base_name + ".fits")
    stacked_image_info_path = stacked_image_dir / pathlib.Path(base_name + ".json")

    # log.info(
    #     "Saving to %s with filter type %s",
    #     stacked_image_path,
    #     filter_str,
    # )

    # turn our non-image data into JSON and write it out
    info_dict = asdict(image)
    info_dict.pop("stacked_image")
    info_dict["sources"] = [
        (str(obs), str(imgpath), extension)
        for (obs, imgpath, extension) in info_dict["sources"]
    ]
    print(info_dict)

    with open(stacked_image_info_path, "w") as f:
        json.dump(info_dict, f)

    # now save the image
    hdu = fits.PrimaryHDU(image.stacked_image)
    hdu.writeto(stacked_image_path)
