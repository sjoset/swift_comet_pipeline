import pathlib
import itertools
import numpy as np

from rich import print as rprint
from rich.panel import Panel

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt

from swift_comet_pipeline.epochs import Epoch
from swift_comet_pipeline.stacking import (
    StackedUVOTImageSet,
    StackingMethod,
    stack_epoch_into_sum_and_median,
)
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
    filter_to_string,
)
from swift_comet_pipeline.tui import (
    bool_to_x_or_check,
    epoch_menu,
    get_yes_no,
    wait_for_key,
)
from swift_comet_pipeline.uvot_image import (
    SwiftUVOTImage,
    get_uvot_image_center,
    pad_to_match_sizes,
)
from swift_comet_pipeline.pipeline_files import (
    PipelineEpochID,
    PipelineFiles,
    PipelineProductType,
)


# TODO: move this into pipeline_files?
def is_epoch_fully_stacked(
    pipeline_files: PipelineFiles, epoch_id: PipelineEpochID
) -> bool:
    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]

    # for filter_type, stacking_method in product(filters, stacking_methods):
    stack_exists = [
        pipeline_files.exists(
            PipelineProductType.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        for filter_type, stacking_method in itertools.product(filters, stacking_methods)
    ]

    return all(stack_exists)


def print_stacked_images_summary(
    pipeline_files: PipelineFiles,
) -> None:
    epoch_ids = pipeline_files.get_epoch_ids()
    if epoch_ids is None:
        return

    print("Summary of detected stacked images:")
    for epoch_id in epoch_ids:
        rprint(
            "\t",
            epoch_id,
            "\t",
            bool_to_x_or_check(
                is_epoch_fully_stacked(pipeline_files=pipeline_files, epoch_id=epoch_id)
            ),
        )


def menu_stack_all_or_selection() -> str:
    user_selection = None

    print("Stack all (a), make a selection (s), or quit? (q)")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or raw_selection == "s" or raw_selection == "q":
            user_selection = raw_selection

    return user_selection


# TODO: move this to pipeline_files?
def epoch_stacked_image_to_fits(epoch: Epoch, img: SwiftUVOTImage) -> fits.ImageHDU:
    hdu = fits.ImageHDU(data=img)

    # TODO: include data mode or event mode here, time of processing, pipeline version?

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


def show_stacked_image_set(stacked_image_set: StackedUVOTImageSet):
    _, axes = plt.subplots(2, 2, figsize=(100, 20))

    zscale = ZScaleInterval()

    filter_types = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    for i, (filter_type, stacking_method) in enumerate(
        itertools.product(filter_types, stacking_methods)
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


def uw1_and_uvv_stacks_from_epoch(
    pipeline_files: PipelineFiles,
    swift_data: SwiftData,
    epoch: Epoch,
    epoch_id: PipelineEpochID,
    do_coincidence_correction: bool,
    ask_to_save_stack: bool,
    show_stacked_images: bool,
) -> None:
    """
    Produces sum- and median-stacked images for the uw1 and uvv filters - the pipeline will overwrite images if the already exist
    The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
    """
    # Do both filters with sum and median stacking
    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]

    if epoch.DATAMODE.nunique() != 1:
        print("Images in the requested stack have mixed data modes!  Exiting.")
        exit(1)

    epoch_pixel_resolution = epoch.DATAMODE[0]
    stacked_images = StackedUVOTImageSet({})

    # do the stacking
    for filter_type in filter_types:
        print(f"Stacking for filter {filter_to_string(filter_type)} ...")

        # now narrow down the data to just one filter at a time
        filter_mask = epoch["FILTER"] == filter_type
        stacked_epoch = epoch[filter_mask]

        stack_result = stack_epoch_into_sum_and_median(
            swift_data=swift_data,
            epoch=stacked_epoch,  # type: ignore
            do_coincidence_correction=do_coincidence_correction,
            pixel_resolution=epoch_pixel_resolution,
        )
        if stack_result is None:
            print("Stacking image failed!")
            return

        stacked_images[(filter_type, StackingMethod.summation)] = stack_result[0]
        stacked_images[(filter_type, StackingMethod.median)] = stack_result[1]

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
        show_stacked_image_set(stacked_images)

    if ask_to_save_stack:
        print("Save results?")
        save_results = get_yes_no()
        if not save_results:
            return

    rprint("[green]Writing stacked epoch ...")
    pipeline_files.write_pipeline_product(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id, data=stacked_epoch
    )

    # write the stacked images as .FITS files
    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        hdu = epoch_stacked_image_to_fits(
            epoch=epoch, img=stacked_images[(filter_type, stacking_method)]
        )
        rprint(
            f"[green]Writing stacked image for {epoch_id}, {filter_type} {stacking_method} ..."
        )
        pipeline_files.write_pipeline_product(
            PipelineProductType.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            data=hdu,
        )


def epoch_stacking_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_ids = pipeline_files.get_epoch_ids()
    if epoch_ids is None:
        print("No epochs found!")
        wait_for_key()
        return

    print_stacked_images_summary(pipeline_files=pipeline_files)

    fully_stacked = [
        is_epoch_fully_stacked(pipeline_files=pipeline_files, epoch_id=x)
        for x in epoch_ids
    ]
    if all(fully_stacked):
        print("Everything stacked! Nothing to do.")
        # TODO: ask to continue anyway
        wait_for_key()
        return

    menu_selection = menu_stack_all_or_selection()
    if menu_selection == "q":
        return

    # do we stack all of the epochs, or just select one?
    epoch_ids_to_stack = []
    ask_to_save_stack = True
    show_stacked_images = True
    skip_if_stacked = False
    if menu_selection == "a":
        epoch_ids_to_stack = pipeline_files.get_epoch_ids()
        ask_to_save_stack = False
        show_stacked_images = False
        skip_if_stacked = True
    elif menu_selection == "s":
        epoch_id_selected = epoch_menu(pipeline_files)
        if epoch_id_selected is None:
            return
        epoch_ids_to_stack = [epoch_id_selected]

    if epoch_ids_to_stack is None:
        print("Pipeline error! This is a bug with epoch_ids_to_stack")
        return

    # for each epoch selected, load the epoch and stack the images in it
    for epoch_id, is_stacked in zip(epoch_ids_to_stack, fully_stacked):
        epoch = pipeline_files.read_pipeline_product(
            PipelineProductType.epoch, epoch_id=epoch_id
        )
        if epoch is None:
            print(f"{epoch_id=} could not be loaded, skipping..")
            continue
        epoch_path = pipeline_files.get_product_path(
            PipelineProductType.epoch, epoch_id=epoch_id
        )
        if epoch_path is None:
            print(f"Path associated with {epoch_id=} was not found! This is a bug.")
            continue

        # filter out the manually vetoed images
        non_vetoed_epoch = epoch[epoch.manual_veto == np.False_]

        # check if the stacked images exist and ask to replace, unless we are stacking all epochs - in that case, skip the stacks we already have
        if is_stacked and skip_if_stacked:
            continue

        rprint(Panel(f"Epoch {epoch_id}:", expand=False))
        uw1_and_uvv_stacks_from_epoch(
            pipeline_files=pipeline_files,
            swift_data=swift_data,
            epoch=non_vetoed_epoch,
            epoch_id=epoch_id,
            do_coincidence_correction=True,
            ask_to_save_stack=ask_to_save_stack,
            show_stacked_images=show_stacked_images,
        )

    wait_for_key()
