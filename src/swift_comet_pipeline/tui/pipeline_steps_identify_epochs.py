from rich import print as rprint

from swift_comet_pipeline.observationlog.gui_observation_log_slicing import (
    gui_select_epoch_time_window,
)
from swift_comet_pipeline.observationlog.slice_observation_log_into_epochs import (
    epochs_from_time_delta,
    split_epoch_list_into_data_and_event_epochs,
)
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps import (
    SwiftCometPipelineStepStatus,
)
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.tui.tui_common import get_yes_no, wait_for_key
from swift_comet_pipeline.observationlog.observation_log import (
    includes_uvv_and_uw1_filters,
)
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


def identify_epochs_step(swift_project_config: SwiftProjectConfig) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    if (
        scp.get_status(SwiftCometPipelineStepEnum.observation_log)
        != SwiftCometPipelineStepStatus.complete
    ):
        rprint("[red]Observation log has not been generated![/red]")
        wait_for_key()
        return

    obs_log = scp.get_product_data(pf=PipelineFilesEnum.observation_log)
    assert obs_log is not None

    if not includes_uvv_and_uw1_filters(obs_log=obs_log):
        rprint(
            "[red]The selection does not have data in both [blue]uw1[/blue] and [purple]uvv[/purple] filters![/red]"
        )

    # only show uw1 and uvv filters on timeline, which means our epochs only include these two filters as well
    filter_mask = (obs_log["FILTER"] == SwiftFilter.uw1) | (
        obs_log["FILTER"] == SwiftFilter.uvv
    )
    filtered_obs_log = obs_log[filter_mask]

    dt = gui_select_epoch_time_window(obs_log=filtered_obs_log)
    epoch_list = epochs_from_time_delta(
        obs_log=filtered_obs_log, max_time_between_obs=dt
    )
    # print(f"Pre-split epochs: {len(epoch_list)}")
    epoch_list = split_epoch_list_into_data_and_event_epochs(epoch_list=epoch_list)
    # print(f"Post-split epochs: {len(epoch_list)}")

    print("Save epochs?")
    save_epochs = get_yes_no()
    if not save_epochs:
        return

    # data_ingestion_files.create_epochs(epoch_list=epoch_list, write_to_disk=True)
    scp.pipeline_files.create_pre_stack_epochs(
        epoch_list=epoch_list, write_to_disk=True
    )

    rprint("[green]Done writing epochs![/green]")
    wait_for_key()
