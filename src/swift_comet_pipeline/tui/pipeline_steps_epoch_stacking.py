from itertools import product

from rich import print as rprint
from rich.panel import Panel
from icecream import ic

from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.stacking.stacked_uvot_image_set import show_stacked_image_set
from swift_comet_pipeline.stacking.stacking import (
    get_stacked_image_set,
    make_uw1_and_uvv_stacks,
    write_uw1_and_uvv_stacks,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.tui.tui_common import (
    bool_to_x_or_check,
    get_yes_no,
)
from swift_comet_pipeline.tui.tui_menus import subpipeline_selection_menu


def print_stacked_images_summary(
    scp: SwiftCometPipeline,
) -> None:

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None
    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    sms = [StackingMethod.summation, StackingMethod.median]

    print("Summary of detected stacked images:")
    for epoch_id in epoch_ids:
        rprint(epoch_id, end="")
        for f, s in product(filters, sms):
            e = scp.exists(
                pf=PipelineFilesEnum.stacked_image,
                epoch_id=epoch_id,
                filter_type=f,
                stacking_method=s,
            )
            rprint(f"\t{f}, {s}: {bool_to_x_or_check(e)}", end="")
        print("")


def menu_stack_all_or_selection() -> str:
    user_selection = None

    print("Stack all (a), make a selection (s), or quit? (q)")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or raw_selection == "s" or raw_selection == "q":
            user_selection = raw_selection

    return user_selection


def epoch_stacking_step(swift_project_config: SwiftProjectConfig) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)
    # pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)
    swift_data = SwiftData(data_path=swift_project_config.swift_data_path)

    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipelines = pipeline_files.epoch_subpipelines

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return
    #
    # if epoch_subpipelines is None:
    #     print("No epochs available to stack!")
    #     return

    print_stacked_images_summary(scp=scp)

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None
    # epoch_has_been_stacked = {epoch_id: scp.has_epoch_been_stacked(epoch_id) for epoch_id in epoch_ids}
    epochs_stacked = [scp.has_epoch_been_stacked(epoch_id) for epoch_id in epoch_ids]

    if all(epochs_stacked):
        print("Everything stacked! Nothing to do.")
        # TODO: ask to continue anyway
        # return

    menu_selection = menu_stack_all_or_selection()
    if menu_selection == "q":
        return

    # do we stack all of the epochs, or just select one?
    epoch_ids_to_stack = None
    # do we automatically save stacked images, or do we ask first?
    ask_to_save_stack = True
    # show a plot with the four image results?
    show_stacked_images = True
    # if we have already stacked an epoch, do we avoid doing it again?
    skip_if_stacked = False

    if menu_selection == "a":
        epoch_ids_to_stack = epoch_ids
        ask_to_save_stack = False
        show_stacked_images = False
        skip_if_stacked = True
    elif menu_selection == "s":
        # parent_epoch_id_selection = epoch_menu(scp)
        parent_epoch_id_selection = subpipeline_selection_menu(
            scp=scp, status_marker=SwiftCometPipelineStepEnum.epoch_stack
        )
        if parent_epoch_id_selection is None:
            return
        epoch_ids_to_stack = [parent_epoch_id_selection]

    if epoch_ids_to_stack is None:
        ic("We somehow have no selection of epochs to stack! This is a bug!")
        return

    # for each parent epoch selected, get the subpipeline and stack the images in it
    # for parent_epoch, is_stacked in zip(epoch_ids_to_stack, fully_stacked):
    for epoch_id_to_stack in epoch_ids_to_stack:

        # # check if the stacked images exist and ask to replace, unless we are stacking all epochs - in that case, skip the stacks we already have
        # if is_stacked and skip_if_stacked:
        #     print(f"Skipping {parent_epoch.epoch_id} - already stacked")
        #     continue

        # check if the stacked images exist and ask to replace, unless we are stacking all epochs - in that case, skip the stacks we already have
        if scp.has_epoch_been_stacked(epoch_id_to_stack) and skip_if_stacked:
            print(f"Skipping {epoch_id_to_stack} - already stacked")
            continue

        # epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        #     parent_epoch=parent_epoch
        # )
        # if epoch_subpipeline is None:
        #     print(f"Could not find subpipeline for epoch {parent_epoch.epoch_id}")
        #     continue

        rprint(Panel(f"Epoch {epoch_id_to_stack}:", expand=False))
        make_uw1_and_uvv_stacks(
            swift_data=swift_data, scp=scp, epoch_id=epoch_id_to_stack
        )

        if show_stacked_images:
            stacked_image_set = get_stacked_image_set(
                scp=scp, epoch_id=epoch_id_to_stack
            )
            if stacked_image_set is not None:
                show_stacked_image_set(
                    stacked_image_set=stacked_image_set, epoch_id=epoch_id_to_stack
                )

        if ask_to_save_stack:
            print(f"Save results for epoch {epoch_id_to_stack}?")
            save_results = get_yes_no()
            if save_results:
                write_uw1_and_uvv_stacks(scp=scp, epoch_id=epoch_id_to_stack)
                print("Done.")
        else:
            write_uw1_and_uvv_stacks(scp=scp, epoch_id=epoch_id_to_stack)
