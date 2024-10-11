from functools import reduce
from astropy.time import Time

import numpy as np
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.observationlog.observation_log import (
    get_image_path_from_obs_log_row,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.tui.tui_common import epoch_menu

# from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles


def pipeline_extra_epoch_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:
    # pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)
    swift_data = SwiftData(data_path=swift_project_config.swift_data_path)

    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     # wait_for_key()
    #     return
    #
    # if epoch_subpipeline_files is None:
    #     print("No epochs available to stack!")
    #     # wait_for_key()
    #     return

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]

    epoch_id_selected = epoch_menu(scp=scp)
    if epoch_id_selected is None:
        return

    epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id_selected
    )
    assert epoch is not None
    # if epoch is None:
    #     print("Error loading epoch!")
    #     return

    # epoch_id_selected.read()
    # epoch = epoch_id_selected.data

    for filter_type in filters:
        epoch_mask = epoch.FILTER == filter_type
        filtered_epoch = epoch[epoch_mask]
        print(f"Observations in filter {filter_to_file_string(filter_type)}:")

        print(filtered_epoch)
        imglist = set(
            [
                str(
                    get_image_path_from_obs_log_row(
                        swift_data=swift_data, obs_log_row=row
                    )
                )
                for _, row in filtered_epoch.iterrows()
            ]
        )
        print("")
        print("Fits images used:")
        for img in imglist:
            print(img)

        print(
            f"Total exposure time in filter {filter_to_file_string(filter_type)}: {np.sum(filtered_epoch.EXPOSURE)}"
        )
        print("")

    # epoch_subpipeline_files = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #     parent_epoch=epoch_id_selected
    # )
    # if epoch_subpipeline_files is None:
    #     return
    # stacked_epoch_product = epoch_subpipeline_files.stacked_epoch
    # if stacked_epoch_product is None:
    #     return

    # if not stacked_epoch_product.exists():
    #     return
    # stacked_epoch_product.read()
    # stacked_epoch = stacked_epoch_product.data
    # if stacked_epoch is None:
    #     return

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    print("-----Stacked epoch-----")
    print(stacked_epoch)


def pipeline_extra_latex_table_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:
    # pipeline_files = PipelineFiles(swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    epoch_id_selected = epoch_menu(scp=scp)
    # epoch_product = epoch_menu(data_ingestion_files=data_ingestion_files)
    # if epoch_product is None:
    #     return
    #
    # epoch_product.read()
    # epoch = epoch_product.data
    # if epoch is None:
    #     print("Error loading epoch!")
    #     return
    epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id_selected
    )
    assert epoch is not None

    print("")
    print("Obs Date & Filter & Images & Exposure Time & R_h & delta")
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        epoch_mask = epoch.FILTER == filter_type
        filtered_epoch = epoch[epoch_mask]

        obs_times = [Time(x) for x in filtered_epoch.MID_TIME]  # type: ignore
        obs_dates = [x.to_datetime().date() for x in obs_times]  # type: ignore

        unique_days = np.unique(obs_dates)
        unique_days_str = reduce(lambda x, y: str(x) + ", " + str(y), unique_days)

        num_images = len(filtered_epoch)
        exposure_time = np.sum(filtered_epoch.EXPOSURE)
        rh = np.mean(filtered_epoch.HELIO)
        delta = np.mean(filtered_epoch.OBS_DIS)

        print(
            f" & {unique_days_str} & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f} & {rh:3.2f} & {delta:3.2f} \\\\"
        )
