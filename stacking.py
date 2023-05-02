#!/usr/bin/env python3

import pathlib
import json
import numpy as np
import logging as log

from astropy.io import fits
from typing import Tuple, Optional, List
from dataclasses import asdict

from swift_types import (
    SwiftData,
    SwiftObservationID,
    SwiftOrbitID,
    SwiftObservationLog,
    SwiftFilter,
    SwiftUVOTImage,
    SwiftStackedUVOTImage,
    PixelCoord,
    filter_to_file_string,
    SwiftPixelResolution,
    SwiftStackingMethod,
)
from image_cleanup import coincidence_correction


__all__ = [
    "get_image_dimensions_to_center_comet",
    "determine_stacking_image_size",
    "center_image_on_coords",
    # "stack_by_obsid",
    # "stack_by_orbits",
    "includes_uvv_and_uw1_filters",
    "stack_image_by_selection",
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
) -> Optional[Tuple[int, int]]:
    """Opens every FITS file specified in the SwiftObservationLog and finds the image size necessary for stacking all of them"""

    # how far in pixels each comet image needs to shift
    image_dimensions_list = []

    for _, row in obs_log.iterrows():
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


def get_stacked_image_base_str(
    contributing_obsids: List[SwiftObservationID],
    filter_type: SwiftFilter,
    stack_method: SwiftStackingMethod,
) -> str:
    """
    Returns a base name for a stacked image based on the filter and the observation ids that contributed to it
    """
    filter_str = filter_to_file_string(filter_type)

    contributing_obsids = sorted(np.unique(contributing_obsids))
    obsids_str = "_".join(contributing_obsids)

    obsids_str = contributing_obsids[0] + "_through_" + contributing_obsids[-1]

    # name the file
    base_name = obsids_str + "_" + filter_str + "_" + stack_method

    return base_name


def get_stacked_image_path(
    stacked_image_dir: pathlib.Path,
    contributing_obsids: List[SwiftObservationID],
    filter_type: SwiftFilter,
    stack_method: SwiftStackingMethod,
) -> pathlib.Path:
    base_name = get_stacked_image_base_str(
        contributing_obsids=contributing_obsids,
        filter_type=filter_type,
        stack_method=stack_method,
    )
    stacked_image_path = stacked_image_dir / pathlib.Path(base_name + ".fits")

    return stacked_image_path


def get_stacked_image_info_path(
    stacked_image_dir: pathlib.Path,
    contributing_obsids: List[SwiftObservationID],
    filter_type: SwiftFilter,
    stack_method: SwiftStackingMethod,
) -> pathlib.Path:
    base_name = get_stacked_image_base_str(
        contributing_obsids=contributing_obsids,
        filter_type=filter_type,
        stack_method=stack_method,
    )
    stacked_image_info_path = stacked_image_dir / pathlib.Path(base_name + ".json")

    return stacked_image_info_path


def write_stacked_image(
    stacked_image_dir: pathlib.Path, image: SwiftStackedUVOTImage
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Takes the stacked image, names it systematically, and writes two files: one FITS image and one for JSON metadata
    Returns a tuple of the path of the written file and the path of the JSON file
    """
    contributing_obsids = [x[0] for x in image.sources]

    # TODO: just pass the image to the function instead
    stacked_image_path = get_stacked_image_path(
        stacked_image_dir=stacked_image_dir,
        contributing_obsids=contributing_obsids,
        filter_type=image.filter_type,
        stack_method=image.stacking_method,
    )

    stacked_image_info_path = get_stacked_image_info_path(
        stacked_image_dir=stacked_image_dir,
        contributing_obsids=contributing_obsids,
        filter_type=image.filter_type,
        stack_method=image.stacking_method,
    )

    # turn our non-image data into JSON and write it out
    info_dict = asdict(image)
    info_dict.pop("stacked_image")
    info_dict["sources"] = [
        (str(obs), str(imgpath), extension)
        for (obs, imgpath, extension) in info_dict["sources"]
    ]

    with open(stacked_image_info_path, "w") as f:
        json.dump(info_dict, f)

    # now save the image
    hdu = fits.PrimaryHDU(image.stacked_image)
    hdu.writeto(stacked_image_path, overwrite=True)

    return (stacked_image_path, stacked_image_info_path)


def includes_uvv_and_uw1_filters(
    obs_log: SwiftObservationLog,
) -> Tuple[bool, List[SwiftOrbitID]]:
    """
    To find OH and perform dust subtraction we need data from the UV and UW1 filter from somewhere across the given data set in orbit_ids.
    Returns a list of orbits that have UV or UW1 images, after removing orbits that have no data in the UV or UW1 filters
    """

    has_uvv_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uvv]
    has_uvv_set = set(has_uvv_filter["ORBIT_ID"])

    has_uw1_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uw1]
    has_uw1_set = set(has_uw1_filter["ORBIT_ID"])

    has_both = len(has_uvv_set & has_uw1_set) > 0

    contributing_orbits = has_uvv_set
    contributing_orbits.update(has_uw1_set)

    return (has_both, list(contributing_orbits))


# # TODO
# def read_stacked_image(
#         stacked_image_dir: pathlib.Path, stacked_image_path: pathlib.Path, stacked_image_info_path: pathlib.Path
# ) -> SwiftStackedUVOTImage:
#     # read the fits file
#     image_data = fits.getdata(stacked_image_dir / stacked_image_path)
#     # load its metadata
#     with open(stack_image_info_path, "r") as f:
#         image_info = json.load(f)


def stack_image_by_selection(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    do_coincidence_correction: bool = True,
    detector_scale: SwiftPixelResolution = SwiftPixelResolution.data_mode,
    stacking_method: SwiftStackingMethod = SwiftStackingMethod.summation,
) -> Optional[SwiftStackedUVOTImage]:
    """
    Takes every entry in the given SwiftObservationLog and attempts to stack it - it will not check if there are mixed filters!
    """

    has_both_filters, _ = includes_uvv_and_uw1_filters(obs_log)
    if has_both_filters:
        print(
            "Requested stacking includes data from mixed filters! Are you sure you know what you're doing?"
        )
        return None

    # determine how big our stacked image needs to be
    stacking_image_size = determine_stacking_image_size(
        swift_data=swift_data,
        obs_log=obs_log,
    )

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    for _, row in obs_log.iterrows():
        log.info(
            "Found extension %s of %s to stack ...",
            row["EXTENSION"],
            row["FITS_FILENAME"],
        )

    image_data_to_stack: List[SwiftUVOTImage] = []
    exposure_times: List[float] = []
    source_filenames: List[pathlib.Path] = []
    source_extensions: List[int] = []

    for _, row in obs_log.iterrows():
        obsid = row["OBS_ID"]

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore

        log.info(
            "Processing %s, extension %s: Comet center at %s, %s",
            image_path.name,
            row["EXTENSION"],
            row["PX"],
            row["PY"],
        )

        source_filenames.append(pathlib.Path(image_path.name))
        source_extensions.append(int(row["EXTENSION"]))

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
            image_data = coincidence_correction(image_data, detector_scale)

        image_data_to_stack.append(image_data)
        exposure_times.append(float(row["EXPOSURE"]))

    exposure_time = np.sum(exposure_times)

    if stacking_method == SwiftStackingMethod.summation:
        stacked_image = np.sum(image_data_to_stack, axis=0)
    elif stacking_method == SwiftStackingMethod.median:
        stacked_image = np.median(image_data_to_stack, axis=0)
    else:
        log.info("Invalid stacking method specified, defaulting to summation...")
        stacked_image = np.sum(image_data_to_stack, axis=0)

    return SwiftStackedUVOTImage(
        stacked_image=stacked_image,
        sources=list(
            zip(
                # duplicate the obsid into a list of the appropriate length
                obs_log["OBS_ID"],
                source_filenames,
                source_extensions,
            )
        ),
        exposure_time=exposure_time,
        filter_type=obs_log["FILTER"].iloc[0],
        coincidence_corrected=do_coincidence_correction,
        detector_scale=detector_scale,
        stacking_method=stacking_method,
    )
