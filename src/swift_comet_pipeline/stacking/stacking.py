from typing import Tuple, List

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from swift_comet_pipeline.comet.comet_center import get_comet_center_prefer_user_coords
from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
    get_image_dimensions_to_center_on_pixel,
)
from swift_comet_pipeline.observationlog.observation_log import (
    get_image_from_obs_log_row,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.uvot_image import (
    SwiftUVOTImage,
    SwiftPixelResolution,
)
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.swift.coincidence_correction import coincidence_correction


def determine_stacking_image_size(
    swift_data: SwiftData,
    epoch: Epoch,
) -> Tuple[int, int] | None:
    """
    Opens every FITS file specified in the given epoch and finds the image size necessary to accommodate
    the largest image involved in the stack, so we can pad out the smaller images and stack them in one step
    """

    # stores how big each image would need to be if recentered on the comet
    recentered_image_dimensions = []

    for _, row in epoch.iterrows():
        image_data = get_image_from_obs_log_row(swift_data=swift_data, obs_log_row=row)

        comet_center_coords = get_comet_center_prefer_user_coords(row=row)
        # keep a list of the image sizes
        image_dimensions = get_image_dimensions_to_center_on_pixel(
            source_image=image_data, coords_to_center=comet_center_coords
        )
        recentered_image_dimensions.append(image_dimensions)

    if len(recentered_image_dimensions) == 0:
        print("No images found in epoch!")
        return None

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(
        recentered_image_dimensions, key=lambda k: k[0], reverse=True
    )[0][0]
    max_num_cols = sorted(
        recentered_image_dimensions, key=lambda k: k[1], reverse=True
    )[0][1]

    return (max_num_rows, max_num_cols)


def stack_epoch_into_sum_and_median(
    swift_data: SwiftData,
    epoch: Epoch,
    do_coincidence_correction: bool = True,
    pixel_resolution: SwiftPixelResolution = SwiftPixelResolution.data_mode,
) -> Tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage] | None:
    """
    Blindly takes every entry in the given Epoch and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    If successful, returns a tuple of images: (sum, median, exposure_map)
    The exposure_map image has pixels with values in units of seconds - the total exposure time from the stack of images involved
    """
    # TODO: if we never use the median stacking mode, we can instead recursively break the epoch in halves and stack what fits in memory at a single time
    # is that functionality that we want to abandon?

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
    exposure_map_list = []

    # TODO: we should return the dataframe of images that actually get used to keep track of what gets omitted
    stacking_progress_bar = tqdm(epoch.iterrows(), total=len(epoch), unit="images")
    for _, row in stacking_progress_bar:
        obsid = row["OBS_ID"]

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore

        exp_time = float(row["EXPOSURE"])

        # read the image
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        if image_data is None:
            print(f"Could not read fits image at {image_path}! Skipping it in stack.")
            continue

        # do we use the horizons data, or did the user manually tell us where the comet is?
        comet_center_coords = get_comet_center_prefer_user_coords(row=row)

        # TODO: check if the comet center is outside the bounds of the image and omit it?
        # print out a warning about it at least
        img_height, img_width = image_data.shape  # type: ignore
        if comet_center_coords.x < 0 or comet_center_coords.x > img_width:
            print(f"Image dimensions ==> width={img_width}\theight={img_height}")
            print(f"Invalid comet x coordinate {comet_center_coords.x}! Skipping.")
            continue
        if comet_center_coords.y < 0 or comet_center_coords.y > img_height:
            print(f"Image dimensions ==> width={img_width}\theight={img_height}")
            print(f"Invalid comet y coordinate {comet_center_coords.y}! Skipping.")
            continue

        # new image with the comet nucleus centered
        image_data = center_image_on_coords(
            source_image=image_data,  # type: ignore
            source_coords_to_center=comet_center_coords,
            stacking_image_size=stacking_image_size,
        )

        # do any processing before stacking
        if do_coincidence_correction:
            # TODO: for large event mode images, this is so slow that it is unusable
            # the correction expects images in count rate, but we are storing the raw images so divide by exposure time here
            coi_map = coincidence_correction(
                img=image_data / exp_time, scale=pixel_resolution
            )
            image_data = image_data * coi_map

        image_data_to_stack.append(image_data)
        exposure_times.append(exp_time)

        dead_pixels = image_data == 0
        good_pix = np.ones_like(image_data) * exp_time
        good_pix[dead_pixels] = 0
        exposure_map_list.append(good_pix)

        stacking_progress_bar.set_description(
            f"{image_path.name} extension {row.EXTENSION}"
        )

    if len(image_data_to_stack) == 0:
        print("No valid stacking data left!")
        return None

    final_exposure_map = np.sum(exposure_map_list, axis=0)

    exposure_time = epoch.EXPOSURE.sum()

    # divide by total exposure time so that pixels are count rates
    stack_sum = np.sum(image_data_to_stack, axis=0) / exposure_time

    # divide each image by its exposure time for each image to be in count rate, then take median
    for img, exp_time in zip(image_data_to_stack, exposure_times):
        img /= exp_time
    stack_median = np.median(image_data_to_stack, axis=0)

    return stack_sum, stack_median, final_exposure_map
