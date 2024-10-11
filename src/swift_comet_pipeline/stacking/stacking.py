from typing import Tuple, List
from itertools import product

import numpy as np
from astropy.io import fits
from tqdm import tqdm
from icecream import ic

from swift_comet_pipeline.comet.comet_center import get_comet_center_prefer_user_coords
from swift_comet_pipeline.image_manipulation.image_pad import pad_to_match_sizes
from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
    get_image_dimensions_to_center_on_pixel,
)
from swift_comet_pipeline.observationlog.observation_log import (
    get_image_from_obs_log_row,
)
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.stacking.stacked_uvot_image_set import StackedUVOTImageSet
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
    filter_to_string,
)
from swift_comet_pipeline.swift.uvot_image import (
    SwiftUVOTImage,
    SwiftPixelResolution,
)
from swift_comet_pipeline.observationlog.epoch import (
    Epoch,
    EpochID,
    epoch_stacked_image_to_fits,
    is_epoch_stackable,
)
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

        # check if the comet center is outside the bounds of the image and omit it
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

    # def is_stackable(self, epoch: Epoch) -> bool:
    #     """
    #     Checks that all uw1 and uvv images in this epoch are taken with the same DATAMODE keyword
    #     """
    #
    #     # count the number of unique datamodes: this has to be 1 if we want to stack
    #     return epoch.DATAMODE.nunique() == 1
    #


# TODO: Priority 1: rewrite this
def make_uw1_and_uvv_stacks(
    swift_data: SwiftData,
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    do_coincidence_correction: bool = True,
    remove_vetoed: bool = True,
) -> None:
    """
    Produces sum- and median-stacked images for the uw1 and uvv filters
    The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
    """

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    # # read the parent epoch's observation log
    # if self.parent_epoch.data is None:
    #     self.parent_epoch.read()
    # pre_veto_epoch = self.parent_epoch.data
    # if pre_veto_epoch is None:
    #     print(f"Could not read epoch {self.parent_epoch.epoch_id}!")
    #     return
    pre_veto_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    assert pre_veto_epoch is not None

    # filter out the manually vetoed images from the epoch dataframe?
    if remove_vetoed:
        post_veto_epoch = pre_veto_epoch[pre_veto_epoch.manual_veto == np.False_]
    else:
        post_veto_epoch = pre_veto_epoch

    # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    if not is_epoch_stackable(epoch=post_veto_epoch):
        print("Images in the requested stack have mixed data modes! Skipping.")
        return
    else:
        print(
            f"All images taken with FITS keyword DATAMODE={post_veto_epoch.DATAMODE.iloc[0].value}, stacking..."
        )

    # now get just the uw1 and uvv images
    stacked_epoch_mask = np.logical_or(
        post_veto_epoch.FILTER == SwiftFilter.uw1,
        post_veto_epoch.FILTER == SwiftFilter.uvv,
    )
    epoch_to_stack = post_veto_epoch[stacked_epoch_mask]

    # now epoch_to_stack has no vetoed images, and only contains uw1 or uvv images

    epoch_pixel_resolution = epoch_to_stack.ARCSECS_PER_PIXEL.iloc[0]
    stacked_images = StackedUVOTImageSet({})
    exposure_maps = {}

    # do the stacking
    for filter_type in uw1_and_uvv:
        print(f"Stacking for filter {filter_to_string(filter_type)} ...")

        # now narrow down the data to just one filter at a time
        filter_mask = epoch_to_stack["FILTER"] == filter_type
        epoch_only_this_filter = epoch_to_stack[filter_mask]

        stack_result = stack_epoch_into_sum_and_median(
            swift_data=swift_data,
            epoch=epoch_only_this_filter,
            do_coincidence_correction=do_coincidence_correction,
            pixel_resolution=epoch_pixel_resolution,
        )
        if stack_result is None:
            ic(
                f"Stacking image for filter {filter_to_file_string(filter_type)} failed!"
            )
            return

        stacked_images[(filter_type, StackingMethod.summation)] = stack_result[0]
        stacked_images[(filter_type, StackingMethod.median)] = stack_result[1]
        exposure_maps[filter_type] = stack_result[2]

    # Adjust the images from each filter to be the same size
    for stacking_method in sum_and_median:
        (uw1_img, uvv_img) = pad_to_match_sizes(
            img_one=stacked_images[(SwiftFilter.uw1, stacking_method)],
            img_two=stacked_images[(SwiftFilter.uvv, stacking_method)],
        )
        stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
        stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

    # Adjust the exposure maps as well so that they stay the same size as the stacked images
    uw1_exp_map, uvv_exp_map = pad_to_match_sizes(
        img_one=exposure_maps[SwiftFilter.uw1],
        img_two=exposure_maps[SwiftFilter.uvv],
    )

    # push all the data into the products for writing later
    # self.stacked_epoch.data = epoch_to_stack
    # for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
    #     hdu = epoch_stacked_image_to_fits(
    #         epoch=epoch_to_stack, img=stacked_images[(filter_type, stacking_method)]
    #     )
    #     self.stacked_images[filter_type, stacking_method].data = hdu
    #
    # self.exposure_map[SwiftFilter.uw1].data = epoch_stacked_image_to_fits(
    #     epoch=epoch_to_stack, img=uw1_exp_map
    # )
    # self.exposure_map[SwiftFilter.uvv].data = epoch_stacked_image_to_fits(
    #     epoch=epoch_to_stack, img=uvv_exp_map
    # )

    # push all the data into the products for writing later
    epoch_post_stack_prod = scp.get_product(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert epoch_post_stack_prod is not None
    epoch_post_stack_prod.data = epoch_to_stack

    for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
        hdu = epoch_stacked_image_to_fits(
            epoch=epoch_to_stack, img=stacked_images[(filter_type, stacking_method)]
        )
        img_prod = scp.get_product(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        assert img_prod is not None
        img_prod.data = hdu

    uw1_exp_map_prod = scp.get_product(
        pf=PipelineFilesEnum.exposure_map,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
    )
    assert uw1_exp_map_prod is not None
    uw1_exp_map_prod.data = epoch_stacked_image_to_fits(
        epoch=epoch_to_stack, img=uw1_exp_map
    )
    uvv_exp_map_prod = scp.get_product(
        pf=PipelineFilesEnum.exposure_map,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
    )
    assert uvv_exp_map_prod is not None
    uvv_exp_map_prod.data = epoch_stacked_image_to_fits(
        epoch=epoch_to_stack, img=uvv_exp_map
    )


def write_uw1_and_uvv_stacks(scp: SwiftCometPipeline, epoch_id: EpochID) -> None:
    """
    Writes the stacked epoch dataframe, along with the four images created during stacking, and exposure map
    This is a separate step so that the stacking results can be viewed before deciding to save or not save the results
    """
    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    # self.stacked_epoch.write()
    stacked_epoch = scp.get_product(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert stacked_epoch is not None
    stacked_epoch.write()

    for f, s in product(uw1_and_uvv, sum_and_median):
        img_prod = scp.get_product(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        assert img_prod is not None
        img_prod.write()

    for f in uw1_and_uvv:
        em_prod = scp.get_product(
            pf=PipelineFilesEnum.exposure_map, epoch_id=epoch_id, filter_type=f
        )
        assert em_prod is not None
        em_prod.write()

    # for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
    #     self.stacked_images[filter_type, stacking_method].write()
    #
    # for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
    #     self.exposure_map[filter_type].write()


def get_stacked_image_set(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> StackedUVOTImageSet | None:
    stacked_image_set = {}

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    # TODO: we should check if any of the data is None and return None if so - we don't have a valid set of stacked images somehow
    for f, s in product(uw1_and_uvv, sum_and_median):
        img_prod = scp.get_product(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        assert img_prod is not None
        img_prod.read_product_if_not_loaded()
        # if self.stacked_images[filter_type, stacking_method].data is None:
        #     self.stacked_images[filter_type, stacking_method].read()

        # the 'data' of the product includes a data.header for the FITS header, and data.data for the numpy image array
        # stacked_image_set[filter_type, stacking_method] = self.stacked_images[
        #     filter_type, stacking_method
        # ].data.data
        assert img_prod.data is not None
        stacked_image_set[f, s] = img_prod.data.data

    return stacked_image_set
