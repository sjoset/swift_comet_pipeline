from itertools import product
from typing import Optional

from rich import print as rprint
from rich.console import Console

from swift_comet_pipeline.pipeline_files import (
    PipelineFiles,
    PipelineEpochID,
    PipelineProductType,
)
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.swift_filter import SwiftFilter

__all__ = [
    "get_float",
    "get_yes_no",
    "get_selection",
    "epoch_menu",
    "stacked_epoch_menu",
    "bool_to_x_or_check",
    "wait_for_key",
    "clear_screen",
]


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


def epoch_menu(pipeline_files: PipelineFiles) -> Optional[PipelineEpochID]:
    """Allows selection of an epoch via a text menu"""
    epoch_ids = pipeline_files.get_epoch_ids()
    if epoch_ids is None:
        print("No epochs available!")
        return None

    selection = get_selection(epoch_ids)
    if selection is None:
        return None

    return epoch_ids[selection]


def stacked_epoch_menu(
    pipeline_files: PipelineFiles,
    require_background_analysis_to_be: Optional[bool] = None,
) -> Optional[PipelineEpochID]:
    """
    Allows selection of a stacked epoch via a text menu, showing only epochs that have been stacked,
    returning a path to the associated epoch that generated the stack, which is how we find products
    associated with that epoch in PipelineFiles

    If require_background_analysis_to_be is set to True, then only epochs with completed background analysis are shown.
    If it is false, only epochs without background analysis are shown.
    If it is None, no filtering occurs
    """
    epoch_ids = pipeline_files.get_epoch_ids()
    if epoch_ids is None:
        print("No epochs available!")
        return None

    filtered_epoch_ids = [
        epoch_id
        for epoch_id in epoch_ids
        if pipeline_files.exists(PipelineProductType.stacked_epoch, epoch_id=epoch_id)
    ]

    if require_background_analysis_to_be is not None:
        filters = [SwiftFilter.uw1, SwiftFilter.uvv]
        stacking_methods = [StackingMethod.summation, StackingMethod.median]

        def all_bgs(x: PipelineEpochID) -> bool:
            return all(
                pipeline_files.exists(
                    PipelineProductType.background_analysis,
                    epoch_id=x,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
                for filter_type, stacking_method in product(filters, stacking_methods)
            )

        filtered_epoch_ids = [
            epoch_id
            for epoch_id in filtered_epoch_ids
            if all_bgs(epoch_id) == require_background_analysis_to_be
        ]

    if len(filtered_epoch_ids) == 0:
        print("No stacked epochs available!")
        return None

    selection = get_selection(filtered_epoch_ids)
    if selection is None:
        return None

    return filtered_epoch_ids[selection]


def wait_for_key(prompt: str = "Press enter to continue") -> None:
    _ = input(prompt)


def clear_screen() -> None:
    console = Console()
    console.clear()
