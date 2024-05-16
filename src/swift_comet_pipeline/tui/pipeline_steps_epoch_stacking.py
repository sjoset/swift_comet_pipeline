from rich import print as rprint
from rich.panel import Panel
from icecream import ic

from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking.stacked_uvot_image_set import show_stacked_image_set
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.tui.tui_common import (
    bool_to_x_or_check,
    epoch_menu,
    get_yes_no,
)


def print_stacked_images_summary(
    pipeline_files: PipelineFiles,
) -> None:
    if pipeline_files.epoch_subpipelines is None:
        # TODO: better error message
        print("No subpipelines found!")
        return

    print("Summary of detected stacked images:")
    for epoch_subpipeline in pipeline_files.epoch_subpipelines:
        rprint(
            "\t",
            epoch_subpipeline.parent_epoch.epoch_id,
            "\t",
            bool_to_x_or_check(epoch_subpipeline.all_images_stacked),
        )


def menu_stack_all_or_selection() -> str:
    user_selection = None

    print("Stack all (a), make a selection (s), or quit? (q)")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or raw_selection == "s" or raw_selection == "q":
            user_selection = raw_selection

    return user_selection


def epoch_stacking_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)
    swift_data = SwiftData(data_path=swift_project_config.swift_data_path)

    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipelines = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipelines is None:
        print("No epochs available to stack!")
        return

    print_stacked_images_summary(pipeline_files=pipeline_files)

    fully_stacked = [x.all_images_stacked for x in epoch_subpipelines]
    if all(fully_stacked):
        print("Everything stacked! Nothing to do.")
        # TODO: ask to continue anyway
        return

    menu_selection = menu_stack_all_or_selection()
    if menu_selection == "q":
        return

    # do we stack all of the epochs, or just select one?
    parent_epochs_to_stack = None
    # do we automatically save stacked images, or do we ask first?
    ask_to_save_stack = True
    # show a plot with the four image results?
    show_stacked_images = True
    # if we have already stacked an epoch, do we avoid doing it again?
    skip_if_stacked = False

    if menu_selection == "a":
        parent_epochs_to_stack = data_ingestion_files.epochs
        ask_to_save_stack = False
        show_stacked_images = False
        skip_if_stacked = True
    elif menu_selection == "s":
        parent_epoch_selection = epoch_menu(data_ingestion_files)
        if parent_epoch_selection is None:
            return
        parent_epochs_to_stack = [parent_epoch_selection]

    if parent_epochs_to_stack is None:
        ic("We somehow have no selection of epochs to stack! This is a bug!")
        return

    # for each parent epoch selected, get the subpipeline and stack the images in it
    for parent_epoch, is_stacked in zip(parent_epochs_to_stack, fully_stacked):
        # check if the stacked images exist and ask to replace, unless we are stacking all epochs - in that case, skip the stacks we already have
        if is_stacked and skip_if_stacked:
            print(f"Skipping {parent_epoch.epoch_id} - already stacked")
            continue

        epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
            parent_epoch=parent_epoch
        )
        if epoch_subpipeline is None:
            print(f"Could not find subpipeline for epoch {parent_epoch.epoch_id}")
            continue

        rprint(Panel(f"Epoch {parent_epoch.epoch_id}:", expand=False))
        epoch_subpipeline.make_uw1_and_uvv_stacks(swift_data=swift_data)

        if show_stacked_images:
            stacked_image_set = epoch_subpipeline.get_stacked_image_set()
            if stacked_image_set is not None:
                show_stacked_image_set(
                    stacked_image_set=stacked_image_set, epoch_id=parent_epoch.epoch_id
                )

        if ask_to_save_stack:
            print(f"Save results for epoch {parent_epoch.epoch_id}?")
            save_results = get_yes_no()
            if save_results:
                epoch_subpipeline.write_uw1_and_uvv_stacks()
                print("Done.")
        else:
            epoch_subpipeline.write_uw1_and_uvv_stacks()
