from typing import Optional

from rich import print as rprint
from rich.console import Console

from swift_comet_pipeline.pipeline.pipeline_products import (
    DataIngestionFiles,
    EpochProduct,
    PipelineFiles,
)


def get_float(prompt: str) -> float:
    user_input = None

    while user_input is None:
        raw_selection = input(prompt)
        try:
            selection = float(raw_selection)
        except ValueError:
            print("Numbers only, please\r")
            selection = None

        if selection is not None:
            user_input = selection

    return user_input


def get_selection(selection_list: list) -> Optional[int]:
    user_selection = None

    while user_selection is None:
        rprint("[red]Selection (q to cancel and exit menu):[/red]")
        for i, element in enumerate(selection_list):
            rprint(f"\t[white]{i}:[/white]\t[blue]{element}[/blue]")

        raw_selection = input()
        if raw_selection == "q":
            return None
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(selection_list)):
            user_selection = selection

    return user_selection


def bool_to_x_or_check(x: bool, rich_text: bool = True):
    if x:
        if rich_text:
            return "[green]✔[/green]"
        else:
            return "✔"
    else:
        if rich_text:
            return "[red]✗[/red]"
        else:
            return "✗"


def get_yes_no() -> bool:
    while True:
        raw_selection = input()
        if raw_selection.lower() in ["y", "yes"]:
            return True
        if raw_selection.lower() in ["n", "no"]:
            return False


def epoch_menu(data_ingestion_files: DataIngestionFiles) -> Optional[EpochProduct]:
    """Allows selection of an epoch via a text menu"""
    if data_ingestion_files.epochs is None:
        print("No epochs available!")
        return None

    selection = get_selection(
        [x.product_path.stem for x in data_ingestion_files.epochs]
    )
    if selection is None:
        return None

    return data_ingestion_files.epochs[selection]


def stacked_epoch_menu(
    pipeline_files: PipelineFiles,
    require_background_analysis_to_exist: bool = False,
    require_background_analysis_to_not_exist: bool = False,
) -> Optional[EpochProduct]:
    """
    Allows selection of a stacked epoch via a text menu, showing only epochs that have been stacked,
    returning a path to the associated epoch that generated the stack, which is how we find products
    associated with that epoch in PipelineFiles

    If require_background_analysis_to_exist is set to True, then only epochs with completed background analysis are shown.
    If it is false, only epochs without background analysis are shown.
    If it is None, no filtering occurs
    """

    parent_epochs = pipeline_files.data_ingestion_files.epochs
    if parent_epochs is None:
        return None

    stacked_epochs = []
    for parent_epoch in parent_epochs:
        if parent_epoch is None:
            continue
        epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
            parent_epoch=parent_epoch
        )
        if epoch_subpipeline is None:
            continue
        if epoch_subpipeline.all_images_stacked:
            # stacked images for this epoch exist - do we require the background analysis to be done as well?
            if (
                require_background_analysis_to_exist
                and not epoch_subpipeline.background_analyses_done
            ):
                continue
            if (
                require_background_analysis_to_not_exist
                and epoch_subpipeline.background_analyses_done
            ):
                continue

            stacked_epochs.append(parent_epoch)

    if len(stacked_epochs) == 0:
        print("No stacked epochs available that meet the conditions:")
        print(
            f"- Stacked images exist\n- {require_background_analysis_to_exist=}\n- {require_background_analysis_to_not_exist=}"
        )
        return None

    selection = get_selection([x.product_path.stem for x in stacked_epochs])
    if selection is None:
        return None

    return stacked_epochs[selection]


def wait_for_key(prompt: str = "Press enter to continue") -> None:
    _ = input(prompt)


def clear_screen() -> None:
    console = Console()
    console.clear()
