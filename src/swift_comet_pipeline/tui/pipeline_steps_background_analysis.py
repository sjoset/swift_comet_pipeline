from itertools import product

import questionary
import numpy as np
from astropy.io import fits
from rich import print as rprint

from swift_comet_pipeline.background.determine_background import determine_background
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.stacking.stacking import get_stacked_image_set
from swift_comet_pipeline.swift.swift_filter_to_string import filter_to_file_string
from swift_comet_pipeline.tui.tui_common import wait_for_key
from swift_comet_pipeline.tui.tui_menus import subpipeline_selection_menu
from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.types.background_result import (
    background_result_to_dict,
    yaml_dict_to_background_result,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig

# TODO: break this into two files


# TODO: replace questionary?
def get_background_method_choice() -> BackgroundDeterminationMethod | None:

    bg_method = questionary.select(
        message="Background method: ",
        choices=BackgroundDeterminationMethod.all_bg_determination_methods(),
    ).ask()

    return bg_method


def determine_background_step(swift_project_config: SwiftProjectConfig) -> None:
    # TODO: document function and add option to re-use the background region if we used manual gui or aperture walkers

    uw1_and_uvv = [SwiftFilter.uw1, SwiftFilter.uvv]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    selected_epoch_id = subpipeline_selection_menu(
        scp=scp, status_marker=SwiftCometPipelineStepEnum.determine_background
    )
    if selected_epoch_id is None:
        return

    if not scp.has_epoch_been_stacked(epoch_id=selected_epoch_id):
        print(f"This epoch hasn't been stacked!")
        return

    stacked_image_set = get_stacked_image_set(scp=scp, epoch_id=selected_epoch_id)
    assert stacked_image_set is not None

    bg_method = get_background_method_choice()
    if bg_method is None:
        return

    # stacked_epoch = scp.get_product_data(
    #     pf=PipelineFilesEnum.epoch_post_stack, epoch_id=selected_epoch_id
    # )
    # assert stacked_epoch is not None

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=selected_epoch_id)
    assert epoch_summary is not None

    for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
        img_data = stacked_image_set[filter_type, stacking_method]

        exposure_map = scp.get_product_data(
            pf=PipelineFilesEnum.exposure_map,
            epoch_id=selected_epoch_id,
            filter_type=filter_type,
        )
        assert exposure_map is not None
        exposure_map = exposure_map.data

        bg_result = determine_background(
            img=img_data,
            exposure_map=exposure_map,
            filter_type=filter_type,
            background_method=bg_method,
            epoch_summary=epoch_summary,
        )
        assert bg_result is not None

        rprint(
            f"[blue]{selected_epoch_id}[/blue]\t[green]{filter_to_file_string(filter_type=filter_type)}[/green]\t[orange]{stacking_method}[/orange]"
        )
        print(f"Background count rate: {bg_result.count_rate_per_pixel}")

        rprint(
            f"[green]Writing background analysis for filter {filter_to_file_string(filter_type)}, stacking method {stacking_method}...[/green]"
        )
        bg_determination = scp.get_product(
            pf=PipelineFilesEnum.background_determination,
            epoch_id=selected_epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        assert bg_determination is not None
        bg_determination.data = background_result_to_dict(bg_result=bg_result)
        bg_determination.write()

    wait_for_key()


# TODO: duplicate definition
def menu_stack_all_or_selection() -> str:
    user_selection = None

    print("Stack all (a), make a selection (s), or quit? (q)")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or raw_selection == "s" or raw_selection == "q":
            user_selection = raw_selection

    return user_selection


def background_subtract_step(swift_project_config: SwiftProjectConfig) -> None:

    uw1_and_uvv = [SwiftFilter.uw1, SwiftFilter.uvv]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None
    epoch_ids_to_bg_sub = None

    menu_selection = menu_stack_all_or_selection()
    if menu_selection == "q":
        return
    elif menu_selection == "a":
        epoch_ids_to_bg_sub = epoch_ids
    elif menu_selection == "s":
        parent_epoch_id_selection = subpipeline_selection_menu(
            scp=scp, status_marker=SwiftCometPipelineStepEnum.background_subtract
        )
        if parent_epoch_id_selection is None:
            return
        epoch_ids_to_bg_sub = [parent_epoch_id_selection]
    assert epoch_ids_to_bg_sub is not None

    for epoch_id_to_bg_sub in epoch_ids_to_bg_sub:

        if not scp.has_epoch_been_stacked(epoch_id=epoch_id_to_bg_sub):
            print(f"Skipping {epoch_id_to_bg_sub} - reason: epoch not stacked")
            continue

        stacked_image_set = get_stacked_image_set(scp=scp, epoch_id=epoch_id_to_bg_sub)
        assert stacked_image_set is not None

        stacked_epoch = scp.get_product_data(
            pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_to_bg_sub
        )
        assert stacked_epoch is not None

        for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):

            img_data = stacked_image_set[filter_type, stacking_method]
            img_header = scp.get_product_data(
                pf=PipelineFilesEnum.stacked_image,
                epoch_id=epoch_id_to_bg_sub,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            assert img_header is not None
            img_header = img_header.header

            bg_result = scp.get_product_data(
                pf=PipelineFilesEnum.background_determination,
                epoch_id=epoch_id_to_bg_sub,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            assert bg_result is not None
            bg_result = yaml_dict_to_background_result(raw_yaml=bg_result)

            rprint(
                f"Processing [blue]{epoch_id_to_bg_sub}[/blue]\t[green]{filter_to_file_string(filter_type=filter_type)}[/green]\t[orange1]{stacking_method}[/orange1]"
            )
            # print(f"Background count rate: {bg_result.count_rate_per_pixel}")

            bg_corrected_img = img_data - bg_result.count_rate_per_pixel.value

            # make a new fits with the background-corrected image, and copy the header information over from the original stacked image
            rprint(
                f"[green]Writing background-subtracted FITS image for filter {filter_to_file_string(filter_type)}, stacking method {stacking_method}...[/green]"
            )
            bg_sub_img = scp.get_product(
                pf=PipelineFilesEnum.background_subtracted_image,
                epoch_id=epoch_id_to_bg_sub,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            assert bg_sub_img is not None
            bg_sub_img.data = fits.ImageHDU(data=bg_corrected_img, header=img_header)
            bg_sub_img.write()

    wait_for_key()
