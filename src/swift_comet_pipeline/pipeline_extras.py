from enum import StrEnum

from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline_extras_epoch_summary import (
    pipeline_extra_epoch_summary,
    pipeline_extra_latex_table_summary,
)
from swift_comet_pipeline.pipeline_extras_status import pipeline_extra_status
from swift_comet_pipeline.tui import clear_screen, get_selection, wait_for_key


class PipelineExtrasMenuEntry(StrEnum):
    pipeline_status = "pipeline status"
    epoch_summary = "epoch summary"
    epoch_latex_observation_log = "observation summary in latex format"

    @classmethod
    def all_extras(cls):
        return [x for x in cls]


def pipeline_extras_menu(swift_project_config: SwiftProjectConfig) -> None:
    exit_menu = False
    extras_menu_entries = PipelineExtrasMenuEntry.all_extras()
    while not exit_menu:
        clear_screen()
        step_selection = get_selection(extras_menu_entries)
        if step_selection is None:
            exit_menu = True
            continue
        step = extras_menu_entries[step_selection]

        if step == PipelineExtrasMenuEntry.pipeline_status:
            pipeline_extra_status(swift_project_config=swift_project_config)
            wait_for_key()
        elif step == PipelineExtrasMenuEntry.epoch_summary:
            pipeline_extra_epoch_summary(swift_project_config=swift_project_config)
            wait_for_key()
        elif step == PipelineExtrasMenuEntry.epoch_latex_observation_log:
            pipeline_extra_latex_table_summary(
                swift_project_config=swift_project_config
            )
            wait_for_key()
        else:
            exit_menu = True
