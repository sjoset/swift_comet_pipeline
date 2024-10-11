from itertools import product
from dataclasses import dataclass

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

from swift_comet_pipeline.observationlog.epoch import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps import (
    SwiftCometPipelineStepStatus,
)
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.tui.tui_common import get_selection


@dataclass
class SCPMenuEntry:
    description: str
    status: SwiftCometPipelineStepStatus


# def get_float(prompt: str) -> float:
#     user_input = None
#
#     while user_input is None:
#         raw_selection = input(prompt)
#         try:
#             selection = float(raw_selection)
#         except ValueError:
#             print("Numbers only, please\r")
#             selection = None
#
#         if selection is not None:
#             user_input = selection
#
#     return user_input
#
#
# def get_selection(selection_list: list) -> int | None:
#     user_selection = None
#
#     while user_selection is None:
#         rprint("[red]Selection (q to cancel and exit menu):[/red]")
#         for i, element in enumerate(selection_list):
#             rprint(f"\t[white]{i}:[/white]\t[blue]{element}[/blue]")
#
#         raw_selection = input()
#         if raw_selection == "q":
#             return None
#         try:
#             selection = int(raw_selection)
#         except ValueError:
#             print("Numbers only, please")
#             selection = -1
#
#         if selection in range(len(selection_list)):
#             user_selection = selection
#
#     return user_selection
#
#
# def bool_to_x_or_check(x: bool, rich_text: bool = True):
#     if x:
#         if rich_text:
#             return "[green]✔[/green]"
#         else:
#             return "✔"
#     else:
#         if rich_text:
#             return "[red]✗[/red]"
#         else:
#             return "✗"


def step_status_to_symbol(
    x: SwiftCometPipelineStepStatus, rich_text: bool = True
) -> str:
    match x:
        case SwiftCometPipelineStepStatus.complete:
            if rich_text:
                return "[green]✔[/green]"
            else:
                return "✔"
        case SwiftCometPipelineStepStatus.not_complete:
            if rich_text:
                return "[red]✗[/red]"
            else:
                return "✗"
        case SwiftCometPipelineStepStatus.partial:
            if rich_text:
                return "[yellow]~[/yellow]"
            else:
                return "~"
        case SwiftCometPipelineStepStatus.invalid:
            if rich_text:
                return "[red]![/red]"
            else:
                return "!"


# def get_yes_no() -> bool:
#     while True:
#         raw_selection = input()
#         if raw_selection.lower() in ["y", "yes"]:
#             return True
#         if raw_selection.lower() in ["n", "no"]:
#             return False


def epoch_menu(scp: SwiftCometPipeline) -> EpochID | None:
    """Allows selection of an epoch via a text menu"""

    if (
        scp.get_status(step=SwiftCometPipelineStepEnum.identify_epochs)
        != SwiftCometPipelineStepStatus.complete
    ):
        print("No epochs available!")
        return None

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None

    selection = get_selection(epoch_ids)
    if selection is None:
        return None

    return epoch_ids[selection]


def filter_by_status(
    scp: SwiftCometPipeline,
    epoch_id_list: list[EpochID],
    step: SwiftCometPipelineStepEnum,
    step_status: SwiftCometPipelineStepStatus,
    filter_type: SwiftFilter | None = None,
    stacking_method: StackingMethod | None = None,
) -> list[EpochID]:

    return list(
        filter(
            lambda x: scp.get_status(
                step=step,
                epoch_id=x,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            == step_status,
            epoch_id_list,
        )
    )


def stacked_epoch_menu(scp: SwiftCometPipeline) -> EpochID | None:
    epoch_id_list = scp.get_epoch_id_list()
    if epoch_id_list is None:
        return None

    stacked_epochs = [x for x in epoch_id_list if scp.has_epoch_been_stacked(x)]
    if len(stacked_epochs) == 0:
        return None

    selection = get_selection(stacked_epochs)
    if selection is None:
        return None

    return stacked_epochs[selection]


# def wait_for_key(prompt: str = "Press enter to continue") -> None:
#     _ = input(prompt)
#
#
# def clear_screen() -> None:
#     console = Console()
#     console.clear()


def scp_menu_selection_plain(menu_entries: list[SCPMenuEntry]) -> SCPMenuEntry | None:

    user_selection = None

    while user_selection is None:
        rprint("[red]Selection (q to cancel and exit menu):[/red]")
        for i, entry in enumerate(menu_entries):
            status_string = step_status_to_symbol(entry.status)
            rprint(
                f"\t[white]{i}:[/white]\t[blue]{entry.description}[/blue] [white][{status_string}[/white]]"
            )

        raw_selection = input()
        if raw_selection == "q":
            return None
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(menu_entries)):
            user_selection = selection

    return menu_entries[user_selection]


def scp_menu_selection(menu_entries: list[SCPMenuEntry]) -> SCPMenuEntry | None:

    user_selection = None
    console = Console()

    while user_selection is None:
        rprint("[red]Selection (q to cancel and exit menu):[/red]")
        menu_renderables = []
        for i, entry in enumerate(menu_entries):
            status_string = step_status_to_symbol(entry.status)
            p1 = f"\t[white]{i}:[/white]\t[blue]{entry.description}[/blue] [white][[/white]{status_string}[white]][/white]"
            menu_renderables.append(Panel(p1, expand=True))

        console.print(Columns(menu_renderables))

        raw_selection = input()
        if raw_selection == "q":
            return None
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(menu_entries)):
            user_selection = selection

    return menu_entries[user_selection]


def subpipeline_selection_menu(
    scp: SwiftCometPipeline, status_marker: SwiftCometPipelineStepEnum
) -> EpochID | None:

    uw1_and_uvv = [SwiftFilter.uw1, SwiftFilter.uvv]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    epoch_ids = scp.get_epoch_id_list()
    if epoch_ids is None:
        return None

    statuses = {}
    match status_marker:
        case SwiftCometPipelineStepEnum.epoch_stack:
            for epoch_id in epoch_ids:
                statuses[epoch_id] = scp._status_list_to_single_status(
                    [
                        scp.get_status(
                            SwiftCometPipelineStepEnum.epoch_stack,
                            epoch_id=epoch_id,
                            filter_type=f,
                            stacking_method=s,
                        )
                        for f, s in product(uw1_and_uvv, sum_and_median)
                    ]
                )
        case SwiftCometPipelineStepEnum.determine_background:
            for epoch_id in epoch_ids:
                statuses[epoch_id] = scp._status_list_to_single_status(
                    [
                        scp.get_status(
                            SwiftCometPipelineStepEnum.determine_background,
                            epoch_id=epoch_id,
                            filter_type=f,
                            stacking_method=s,
                        )
                        for f, s in product(uw1_and_uvv, sum_and_median)
                    ]
                )
        case SwiftCometPipelineStepEnum.background_subtract:
            for epoch_id in epoch_ids:
                statuses[epoch_id] = scp._status_list_to_single_status(
                    [
                        scp.get_status(
                            SwiftCometPipelineStepEnum.background_subtract,
                            epoch_id=epoch_id,
                            filter_type=f,
                            stacking_method=s,
                        )
                        for f, s in product(uw1_and_uvv, sum_and_median)
                    ]
                )
        case SwiftCometPipelineStepEnum.aperture_analysis:
            for epoch_id in epoch_ids:
                statuses[epoch_id] = scp._status_list_to_single_status(
                    [
                        scp.get_status(
                            SwiftCometPipelineStepEnum.aperture_analysis,
                            epoch_id=epoch_id,
                            stacking_method=s,
                        )
                        for s in sum_and_median
                    ]
                )
        case SwiftCometPipelineStepEnum.vectorial_analysis:
            for epoch_id in epoch_ids:
                statuses[epoch_id] = scp._status_list_to_single_status(
                    [
                        scp.get_status(
                            SwiftCometPipelineStepEnum.vectorial_analysis,
                            epoch_id=epoch_id,
                            filter_type=f,
                            stacking_method=s,
                        )
                        for f, s in product(uw1_and_uvv, sum_and_median)
                    ]
                )

    menu_entries = [
        SCPMenuEntry(description=epoch_id, status=statuses[epoch_id])
        for epoch_id in epoch_ids
    ]
    selected = scp_menu_selection(menu_entries=menu_entries)

    if selected is None:
        return None
    return selected.description
