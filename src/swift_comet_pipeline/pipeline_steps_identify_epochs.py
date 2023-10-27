from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline_files import EpochProduct, PipelineFiles
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.tui import get_yes_no
from swift_comet_pipeline.observation_log import includes_uvv_and_uw1_filters
from swift_comet_pipeline.epoch_time_window import (
    epochs_from_time_delta,
    select_epoch_time_window,
)


__all__ = ["identify_epochs_step"]


def identify_epochs_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    pipeline_files.observation_log.load_product()
    obs_log = pipeline_files.observation_log.data_product

    if not includes_uvv_and_uw1_filters(obs_log=obs_log):
        print("The selection does not have data in both uw1 and uvv filters!")

    # only show uw1 and uvv filters on timeline
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

    if pipeline_files.epoch_products is not None:
        print("Previous epochs found!  Delete all epochs and overwrite?")
        delete_and_overwrite = get_yes_no()
        if not delete_and_overwrite:
            return
        pipeline_files.delete_epochs_and_their_results()

    epoch_path_list = pipeline_files.determine_epoch_file_paths(epoch_list=epoch_list)
    for epoch, epoch_path in zip(epoch_list, epoch_path_list):
        epoch_product = EpochProduct(product_path=epoch_path)
        epoch_product.data_product = epoch
        print(f"Writing {epoch_product.product_path} ...")
        epoch_product.save_product()
