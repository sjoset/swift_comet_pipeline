from rich import print as rprint

from swift_comet_pipeline.observationlog.gui_observation_log_slicing import (
    gui_select_epoch_time_window,
)
from swift_comet_pipeline.observationlog.slice_observation_log_into_epochs import (
    epochs_from_time_delta,
)
from swift_comet_pipeline.pipeline.files.data_ingestion_files import DataIngestionFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.tui.tui_common import get_yes_no, wait_for_key
from swift_comet_pipeline.observationlog.observation_log import (
    includes_uvv_and_uw1_filters,
)


def identify_epochs_step(swift_project_config: SwiftProjectConfig) -> None:
    data_ingestion_files = DataIngestionFiles(
        project_path=swift_project_config.project_path
    )

    data_ingestion_files.observation_log.read()
    obs_log = data_ingestion_files.observation_log.data
    if obs_log is None:
        rprint("[red]Observation log is missing![/red]")
        wait_for_key()
        return

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

    print("Save epochs?")
    save_epochs = get_yes_no()
    if not save_epochs:
        return

    if data_ingestion_files.epochs is not None:
        print("Found previous epochs in project! Delete epochs and their results?")
        delete_epochs_and_results = get_yes_no()
        # TODO: delete epochs and all of their results
        print("ignoring response for now")

    data_ingestion_files.create_epochs(epoch_list=epoch_list, write_to_disk=True)

    rprint("[green]Done writing epochs![/green]")
    wait_for_key()
