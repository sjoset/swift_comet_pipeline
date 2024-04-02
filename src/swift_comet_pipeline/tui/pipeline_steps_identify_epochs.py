from rich import print as rprint

from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline.pipeline_files import (
    PipelineFiles,
    PipelineProductType,
)
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.tui.tui_common import get_yes_no, wait_for_key
from swift_comet_pipeline.observationlog.observation_log import (
    includes_uvv_and_uw1_filters,
)
from swift_comet_pipeline.pipeline.epoch_time_window import (
    epochs_from_time_delta,
    select_epoch_time_window,
)


__all__ = ["identify_epochs_step"]


def identify_epochs_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    obs_log = pipeline_files.read_pipeline_product(PipelineProductType.observation_log)
    if obs_log is None:
        rprint("[red]Observation is missing![/red]")
        wait_for_key()
        return

    num_epoch_ids = pipeline_files.get_epoch_ids()
    if num_epoch_ids is not None:
        epochs_already_exist = True
    else:
        epochs_already_exist = False

    if not includes_uvv_and_uw1_filters(obs_log=obs_log):
        rprint(
            "[red]The selection does not have data in both [blue]uw1[/blue] and [purple]uvv[/purple] filters![/red]"
        )

    # only show uw1 and uvv filters on timeline, which means our epochs only include these two filters as well
    filter_mask = (obs_log["FILTER"] == SwiftFilter.uw1) | (
        obs_log["FILTER"] == SwiftFilter.uvv
    )
    filtered_obs_log = obs_log[filter_mask]

    dt = select_epoch_time_window(obs_log=filtered_obs_log)
    epoch_list = epochs_from_time_delta(
        obs_log=filtered_obs_log, max_time_between_obs=dt
    )

    print("Save epochs?")
    save_epochs = get_yes_no()
    if not save_epochs:
        return

    if epochs_already_exist:
        print("Previous epochs found!  Delete all epochs and overwrite?")
        delete_and_overwrite = get_yes_no()
        if not delete_and_overwrite:
            return
        pipeline_files.delete_epochs_and_their_results()

    pipeline_files.create_epochs(epoch_list=epoch_list)
    rprint("[green]Done writing epochs![/green]")
    wait_for_key()
