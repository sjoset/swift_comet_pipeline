import pathlib
from typing import Optional

from rich import print as rprint
from rich.console import Console

from swift_comet_pipeline.pipeline_files import PipelineFiles, EpochProduct

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


def epoch_menu(pipeline_files: PipelineFiles) -> Optional[EpochProduct]:
    """Allows selection of an epoch via a text menu"""
    if pipeline_files.epoch_products is None:
        print("No epochs available!")
        return None

    epoch_path_list = [x.product_path for x in pipeline_files.epoch_products]
    selection = get_selection([x.stem for x in epoch_path_list])
    if selection is None:
        return None

    return pipeline_files.epoch_products[selection]


def stacked_epoch_menu(pipeline_files: PipelineFiles) -> Optional[pathlib.Path]:
    """
    Allows selection of a stacked epoch via a text menu, showing only epochs that have been stacked,
    returning a path to the associated epoch that generated the stack, which is how we find products
    associated with that epoch in PipelineFiles
    """
    if pipeline_files.epoch_products is None:
        return None

    epoch_paths = [x.product_path for x in pipeline_files.epoch_products]
    stacked_epoch_paths = [
        pipeline_files.stacked_epoch_products[x].product_path for x in epoch_paths  # type: ignore
    ]

    # filter epochs out of the list if we haven't stacked it by seeing if the stacked_epoch_path exists or not
    filtered_epochs = list(
        filter(lambda x: x[1].exists(), zip(epoch_paths, stacked_epoch_paths))
    )

    selectable_epochs = [x[0] for x in filtered_epochs]
    if len(selectable_epochs) == 0:
        return None
    selection = get_selection(selectable_epochs)
    if selection is None:
        return None

    return selectable_epochs[selection]


def wait_for_key(prompt: str = "Press enter to continue") -> None:
    _ = input(prompt)


def clear_screen() -> None:
    console = Console()
    console.clear()
