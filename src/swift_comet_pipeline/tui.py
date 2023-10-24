import pathlib
from typing import Optional

from swift_comet_pipeline.pipeline_files import PipelineFiles, EpochProduct

__all__ = [
    "get_float",
    "get_yes_no",
    "get_selection",
    "epoch_menu",
    "stacked_epoch_menu",
    "bool_to_x_or_check",
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


def get_selection(selection_list: list) -> int:
    user_selection = None

    while user_selection is None:
        print("Selection:")
        for i, element in enumerate(selection_list):
            print(f"{i}:\t{element}")

        raw_selection = input()
        try:
            selection = int(raw_selection)
        except ValueError:
            print("Numbers only, please")
            selection = -1

        if selection in range(len(selection_list)):
            user_selection = selection

    return user_selection


def bool_to_x_or_check(x: bool):
    if x:
        return "✔"
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
        return None

    epoch_path_list = [x.product_path for x in pipeline_files.epoch_products]
    selection = get_selection([x.stem for x in epoch_path_list])

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
    return selectable_epochs[selection]
