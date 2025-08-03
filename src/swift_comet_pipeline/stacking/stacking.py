from itertools import product

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from icecream import ic

from swift_comet_pipeline.comet.comet_center import get_comet_center_prefer_user_coords
from swift_comet_pipeline.image_manipulation.event_mode_downsample import (
    downsample_event_mode_image,
)
from swift_comet_pipeline.image_manipulation.image_pad import pad_to_match_sizes
from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
)
from swift_comet_pipeline.observationlog.epoch_typing import Epoch, EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import (
    get_unstacked_epoch_summary,
)
from swift_comet_pipeline.stacking.determine_stack_size import (
    determine_stacking_image_size_from_stackables,
)
from swift_comet_pipeline.stacking.event_mode import (
    event_mode_fits_to_time_binned_image,
)
from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter_to_string import (
    filter_to_file_string,
)
from swift_comet_pipeline.observationlog.epoch import epoch_stacked_image_to_fits
from swift_comet_pipeline.swift.coincidence_correction import coincidence_correction
from swift_comet_pipeline.types.stacked_uvot_image_set import StackedUVOTImageSet
from swift_comet_pipeline.types.stacking import (
    StackableUVOTImage,
    StackableUVOTImagePrecursor,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def epoch_row_to_stacking_precursor(
    row: pd.Series, horizons_id: str
) -> StackableUVOTImagePrecursor:

    comet_center = get_comet_center_prefer_user_coords(row=row)
    exposure_time_s = row.EXPOSURE
    data_mode = row.DATAMODE
    img = fits.getdata(filename=row.FULL_FITS_PATH, ext=row.EXTENSION)
    assert img is not None
    hdr = fits.getheader(filename=row.FULL_FITS_PATH, ext=row.EXTENSION)
    assert hdr is not None

    precursor = StackableUVOTImagePrecursor(
        img_hdr=hdr,
        img=img,  # type: ignore
        comet_center=comet_center,
        exposure_time_s=exposure_time_s,
        data_mode=data_mode,
        horizons_id=horizons_id,
    )
    return precursor


def process_stackable_precursor(
    precursor: StackableUVOTImagePrecursor, do_coincidence_correction: bool
) -> StackableUVOTImage:
    """
    For data mode images, applies coincidence correction

    For event mode images: bin by time, coincidence correct, and stack slices
    """

    if precursor.data_mode == SwiftImageMode.data_mode:
        assert isinstance(precursor.img, SwiftUVOTImage)
        if do_coincidence_correction:
            coi_map = coincidence_correction(
                img=precursor.img, scale=SwiftPixelResolution.data_mode
            )
            cc_img = coi_map * precursor.img
        else:
            cc_img = precursor.img
        stackable = StackableUVOTImage(
            img=cc_img,
            comet_center=precursor.comet_center,
            exposure_time_s=precursor.exposure_time_s,
            data_mode=SwiftImageMode.data_mode,
        )
        # TODO: we throw away the exposure mask from the event mode stack - probably fine as the offsets between sub-slices are small
    else:
        binning_result = event_mode_fits_to_time_binned_image(
            precursor_img=precursor,
            num_time_slices=3,
            do_coincidence_correction=do_coincidence_correction,
        )
        stackable = binning_result.sum

    return stackable


def downsample_event_mode_stackable_image(s: StackableUVOTImage) -> StackableUVOTImage:
    """
    Requires an event mode image to be centered on the comet
    """

    # this downsampling preserves the centering on the comet
    downsampled_img = downsample_event_mode_image(img=s.img)

    downsampled_stackable = StackableUVOTImage(
        img=downsampled_img,
        comet_center=get_uvot_image_center(downsampled_img),
        exposure_time_s=s.exposure_time_s,
        data_mode=SwiftImageMode.data_mode,
    )

    return downsampled_stackable


def uniform_pixel_resolution(
    imgs: list[StackableUVOTImage],
) -> list[StackableUVOTImage]:
    """
    Take a list and down-sample all event-mode images so that every image has pixel scale of 1 arcsecond
    """

    new_imgs = []
    for i in imgs:
        if i.data_mode == SwiftImageMode.data_mode:
            new_imgs.append(i)
        else:
            new_imgs.append(downsample_event_mode_stackable_image(i))

    return new_imgs


def stack_images(
    stackable_images: list[StackableUVOTImage],
    stacking_image_final_size_rows_cols: tuple[int, int],
) -> tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage] | None:

    if len(stackable_images) == 0:
        return None

    # TODO: selectively exclude comets if they fall outside the image bounds!

    # # check if the comet center is outside the bounds of the image and omit it
    # img_height, img_width = image_data.shape  # type: ignore
    # if comet_center_coords.x < 0 or comet_center_coords.x > img_width:
    #     print(f"Image dimensions ==> width={img_width}\theight={img_height}")
    #     print(f"Invalid comet x coordinate {comet_center_coords.x}! Skipping.")
    #     continue
    # if comet_center_coords.y < 0 or comet_center_coords.y > img_height:
    #     print(f"Image dimensions ==> width={img_width}\theight={img_height}")
    #     print(f"Invalid comet y coordinate {comet_center_coords.y}! Skipping.")
    #     continue

    final_img_size = stacking_image_final_size_rows_cols

    print("Resizing images ...  ", end="")
    resized_images_to_stack = [
        center_image_on_coords(
            s.img,
            source_coords_to_center=s.comet_center,
            stacking_image_size=final_img_size,
            # show_resulting_image=True,
        )
        for s in stackable_images
    ]

    exposure_times = [s.exposure_time_s for s in stackable_images]

    print("Calculating exposure map ...  ", end="")
    exposure_map_list = []
    for resized_img, exp_time in zip(resized_images_to_stack, exposure_times):
        dead_pixels = resized_img == 0
        good_pix = np.ones_like(resized_img) * exp_time
        good_pix[dead_pixels] = 0
        exposure_map_list.append(good_pix)

    final_exposure_map = np.sum(exposure_map_list, axis=0)
    total_exposure_time_s = np.sum(exposure_times)

    print("Calculating sum stacks ...")
    stack_sum = np.sum(resized_images_to_stack, axis=0) / total_exposure_time_s

    print("Calculating median stacks ...")
    stack_median = np.median(
        [
            img / exp_time_s
            for img, exp_time_s in zip(resized_images_to_stack, exposure_times)
        ],
        axis=0,
    )

    print("Done stacking.")

    return stack_sum, stack_median, final_exposure_map


def stack_epoch_into_sum_and_median(
    epoch: Epoch,
    horizons_id: str,
    do_coincidence_correction: bool,
) -> tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage] | None:
    """
    Blindly takes every entry in the given Epoch and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    If successful, returns a tuple of images: (sum, median, exposure_map)
    The exposure_map image has pixels with values in units of seconds - the total exposure time from the stack of images involved
    """

    event_mode_epoch = epoch[epoch.DATAMODE == SwiftImageMode.event_mode].copy()
    data_mode_epoch = epoch[epoch.DATAMODE == SwiftImageMode.data_mode].copy()

    print(
        f"Event mode images: {len(event_mode_epoch)}\t\tData mode images: {len(data_mode_epoch)}"
    )

    print("Creating precursors ...  ", end="")
    stacking_precursors = [
        epoch_row_to_stacking_precursor(row=row, horizons_id=horizons_id)
        for _, row in tqdm(epoch.iterrows())
    ]

    print("Processing precursors ...  ", end="")
    stackable_images = [
        process_stackable_precursor(
            p, do_coincidence_correction=do_coincidence_correction
        )
        for p in stacking_precursors
    ]

    print("Applying uniform resolution sampling ...  ", end="")
    stackable_images = uniform_pixel_resolution(stackable_images)

    print("Determining final stacked image size ...  ")
    stacking_image_size = determine_stacking_image_size_from_stackables(
        stackable_images
    )

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    stack_results = stack_images(
        stackable_images=stackable_images,
        stacking_image_final_size_rows_cols=stacking_image_size,
    )

    if stack_results is None:
        print("Could not finalize stack! Not stacking.")
        return None

    return stack_results


# TODO: Priority 1: rewrite this
def make_uw1_and_uvv_stacks(
    # swift_data: SwiftData,
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    horizons_id: str,
    do_coincidence_correction: bool = True,
    remove_vetoed: bool = True,
) -> None:
    """
    Produces sum- and median-stacked images for the uw1 and uvv filters
    The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
    """

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    pre_veto_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    assert pre_veto_epoch is not None

    # filter out the manually vetoed images from the epoch dataframe?
    if remove_vetoed:
        post_veto_epoch = pre_veto_epoch[pre_veto_epoch.manual_veto == np.False_]
    else:
        post_veto_epoch = pre_veto_epoch

    # # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    # if not is_epoch_stackable(epoch=post_veto_epoch):
    #     print("Images in the requested stack have mixed data imaging modes")
    # else:
    #     print(
    #         f"All images taken with FITS keyword DATAMODE={post_veto_epoch.DATAMODE.iloc[0].value}, stacking..."
    #     )

    # now get just the uw1 and uvv images
    stacked_epoch_mask = np.logical_or(
        post_veto_epoch.FILTER == SwiftFilter.uw1,
        post_veto_epoch.FILTER == SwiftFilter.uvv,
    )
    epoch_to_stack = post_veto_epoch[stacked_epoch_mask]

    # # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    # if not is_epoch_stackable(epoch=epoch_to_stack):
    #     print("Images in the requested stack have mixed data modes!")
    #     num_event_mode_imgs = epoch_to_stack.DATAMODE.value_counts()[
    #         SwiftImageMode.event_mode
    #     ]
    #     num_data_mode_imgs = epoch_to_stack.DATAMODE.value_counts()[
    #         SwiftImageMode.data_mode
    #     ]
    #     print(
    #         f"Event mode images: {num_event_mode_imgs}\tData mode images: {num_data_mode_imgs}"
    #     )
    #     if num_event_mode_imgs > num_data_mode_imgs:
    #         selected_data_mode = SwiftImageMode.event_mode
    #     else:
    #         selected_data_mode = SwiftImageMode.data_mode
    #     print(f"Filtering images to use only {selected_data_mode.value} ...")
    #     epoch_to_stack = epoch_to_stack[
    #         epoch_to_stack.DATAMODE == selected_data_mode
    #     ].copy()
    # else:
    #     print(
    #         f"All images taken with FITS keyword DATAMODE={epoch_to_stack.DATAMODE.iloc[0].value}, stacking..."
    #     )

    # now epoch_to_stack has no vetoed images, and only contains uw1 or uvv images

    # epoch_pixel_resolution = epoch_to_stack.ARCSECS_PER_PIXEL.iloc[0]
    stacked_images = StackedUVOTImageSet({})
    exposure_maps = {}

    # do the stacking
    for filter_type in uw1_and_uvv:
        print(f"Stacking for filter {filter_to_file_string(filter_type)} ...")

        # now narrow down the data to just one filter at a time
        filter_mask = epoch_to_stack["FILTER"] == filter_type
        epoch_only_this_filter = epoch_to_stack[filter_mask]

        stack_result = stack_epoch_into_sum_and_median(
            epoch=epoch_only_this_filter,
            horizons_id=horizons_id,
            do_coincidence_correction=do_coincidence_correction,
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
    epoch_post_stack_prod = scp.get_product(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert epoch_post_stack_prod is not None
    epoch_post_stack_prod.data = epoch_to_stack

    epoch_summary = get_unstacked_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert epoch_summary is not None
    for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
        hdu = epoch_stacked_image_to_fits(
            epoch_summary=epoch_summary,
            img=stacked_images[(filter_type, stacking_method)],
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
        epoch_summary=epoch_summary, img=uw1_exp_map
    )
    uvv_exp_map_prod = scp.get_product(
        pf=PipelineFilesEnum.exposure_map,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
    )
    assert uvv_exp_map_prod is not None
    uvv_exp_map_prod.data = epoch_stacked_image_to_fits(
        epoch_summary=epoch_summary, img=uvv_exp_map
    )


def write_uw1_and_uvv_stacks(scp: SwiftCometPipeline, epoch_id: EpochID) -> None:
    """
    Writes the stacked epoch dataframe, along with the four images created during stacking, and exposure map
    This is a separate step so that the stacking results can be viewed before deciding to save or not save the results
    This assumes that the stacked images are stored in the SwiftCometPipeline object, ready for writing to file
    """
    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

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


def get_stacked_image_set(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> StackedUVOTImageSet | None:
    stacked_image_set = {}

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    for f, s in product(uw1_and_uvv, sum_and_median):
        img_data = scp.get_product_data(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        if img_data is None:
            return None
        # img_data includes img_data.header for the FITS header, and img_data.data for the numpy image array
        if img_data.data is None:
            return None
        stacked_image_set[f, s] = img_data.data

    return stacked_image_set
