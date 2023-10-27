from functools import reduce
from astropy.time import Time

import numpy as np
from swift_comet_pipeline.configs import SwiftProjectConfig

from swift_comet_pipeline.observation_log import get_image_path_from_obs_log_row
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.tui import epoch_menu


def pipeline_extra_epoch_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    swift_data = SwiftData(swift_project_config.swift_data_path)

    epoch_product = epoch_menu(pipeline_files)
    if epoch_product is None:
        return
    epoch_product.load_product()
    epoch = epoch_product.data_product

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
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


def pipeline_extra_latex_table_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    epoch_product = epoch_menu(pipeline_files)
    if epoch_product is None:
        return
    epoch_product.load_product()
    epoch = epoch_product.data_product

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
