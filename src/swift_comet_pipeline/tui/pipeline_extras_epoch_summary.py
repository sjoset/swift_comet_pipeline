from functools import reduce

import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.epoch_summary import get_epoch_summary
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.observationlog.observation_log import (
    get_image_path_from_obs_log_row,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter_to_string import filter_to_file_string
from swift_comet_pipeline.tui.tui_common import wait_for_key
from swift_comet_pipeline.tui.tui_menus import epoch_menu
from swift_comet_pipeline.types.swift_filter import SwiftFilter


def get_sto(scp: SwiftCometPipeline, epoch_id: EpochID) -> float:
    # Returns the sun-target-observer angle of the given epoch in degrees, or NaN on error
    es = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if es is None:
        return np.nan

    hor = Horizons(
        id=scp.spc.jpl_horizons_id,
        location="@swift",
        epochs=Time(es.observation_time).jd,
    )
    eph = hor.ephemerides()  # type: ignore
    e_df = eph.to_pandas()

    return e_df.alpha[0]


def epoch_id_to_latex_header():
    col_titles = """Epoch & Mid Time & \\(r_h\\) & \\(\\dot{r_h}\\) & \\(\\Delta\\) & S-T-O & \\(T - T_p\\) & Filter & Images & Exposure Time \\\\\n"""
    col_units = """ & & (AU) & (km \\(s^{-1}\\)) & (AU) & (\\(\\degree\\)) & (days) & & & (s)\\\\\n\\hline\\\\"""
    return col_titles + col_units


# TODO: fix this - broken after pipeline rewrite

# def epoch_id_to_latex(scp: SwiftCometPipeline, epoch_id: EpochID) -> None:
#     es = get_epoch_summary(scp=scp, epoch_id=epoch_id)
#     if es is None:
#         return
#
#     stacked_epoch = scp.get_product_data(
#         pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
#     )
#     if stacked_epoch is None:
#         return
#
#     for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
#         epoch_mask = stacked_epoch.FILTER == filter_type
#         filtered_epoch = stacked_epoch[epoch_mask]
#
#         obs_times = [Time(x) for x in filtered_epoch.MID_TIME]
#         obs_dates = [x.to_datetime().date() for x in obs_times]
#
#         unique_days = np.unique(obs_dates)
#         unique_days_str = reduce(lambda x, y: str(x) + ", " + str(y), unique_days)
#         epoch_num = epoch_id_to_epoch_num(epoch_id) + 1
#
#         num_images = len(filtered_epoch)
#         exposure_time = np.sum(filtered_epoch.EXPOSURE)
#         rh = np.mean(filtered_epoch.HELIO)
#         delta = np.mean(filtered_epoch.OBS_DIS)
#         sto = get_sto(scp=scp, epoch_id=epoch_id)
#         tfp = es.time_from_perihelion.to_value(u.day)
#         rdot = es.helio_v_kms
#
#         if filter_type == SwiftFilter.uw1:
#             print(
#                 f" {epoch_num} & {unique_days_str} & {rh:3.2f} & {rdot:3.1f} & {delta:3.2f} & {sto:4.2f} & {tfp:4.1f} & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f}\\\\"
#             )
#         else:
#             print(
#                 f" & & & & & & & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f}\\\\"
#             )


def pipeline_extra_epoch_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)
    swift_data = SwiftData(data_path=swift_project_config.swift_data_path)

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]

    epoch_id_selected = epoch_menu(scp=scp)
    if epoch_id_selected is None:
        return

    epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id_selected
    )
    assert epoch is not None

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

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    print("-----Stacked epoch-----")
    print(stacked_epoch)


def pipeline_extra_latex_table_summary(
    swift_project_config: SwiftProjectConfig,
) -> None:

    # TODO: switch to above function for reporting the entire set of epochs instead of one at a time, using epoch_summary()

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    epoch_id_selected = epoch_menu(scp=scp)
    if epoch_id_selected is None:
        return
    epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id_selected
    )
    assert epoch is not None

    print("Reminder: there is a newer function for this that needs rewriting!")

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

    wait_for_key()
