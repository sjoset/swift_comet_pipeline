from functools import reduce
from astropy.time import Time

from astroquery.jplhorizons import Horizons
import numpy as np
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.epoch_summary import get_epoch_summary
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.observationlog.observation_log import (
    get_image_path_from_obs_log_row,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string


# from swift_comet_pipeline.tui.tui_common import epoch_menu


def epochs_summary(scp: SwiftCometPipeline) -> None:

    epoch_ids = scp.get_epoch_id_list()
    if epoch_ids is None:
        return None

    print(epoch_id_to_latex_header())
    for epoch_id in epoch_ids:
        epoch_id_to_latex(scp=scp, epoch_id=epoch_id)
        print("")


def get_sto(scp: SwiftCometPipeline, epoch_id: EpochID):
    # TODO: handle the None case below
    es = get_epoch_summary(scp=scp, epoch_id=epoch_id)

    hor = Horizons(
        id=scp.spc.jpl_horizons_id,
        location="@swift",
        epochs=Time(es.observation_time).jd,
    )
    eph = hor.ephemerides()
    e_df = eph.to_pandas()

    return e_df.alpha[0]


def epoch_id_to_latex_header():
    col_titles = """Epoch & Mid Time & \\(r_h\\) & \\(\\dot{r_h}\\) & \\(\\Delta\\) & S-T-O & \\(T - T_p\\) & Filter & Images & Exposure Time \\\\\n"""
    col_units = """ & & (AU) & (km \\(s^{-1}\\)) & (AU) & (\\(\\degree\\)) & (days) & & & (s)\\\\\n\\hline\\\\"""
    return col_titles + col_units


def epoch_id_to_latex(scp: SwiftCometPipeline, epoch_id: EpochID):
    # TODO: handle None case
    es = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return

    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        epoch_mask = stacked_epoch.FILTER == filter_type
        filtered_epoch = stacked_epoch[epoch_mask]

        obs_times = [Time(x) for x in filtered_epoch.MID_TIME]
        obs_dates = [x.to_datetime().date() for x in obs_times]

        unique_days = np.unique(obs_dates)
        unique_days_str = reduce(lambda x, y: str(x) + ", " + str(y), unique_days)
        epoch_num = epoch_id_to_epoch_num(epoch_id) + 1

        num_images = len(filtered_epoch)
        exposure_time = np.sum(filtered_epoch.EXPOSURE)
        rh = np.mean(filtered_epoch.HELIO)
        delta = np.mean(filtered_epoch.OBS_DIS)
        sto = get_sto(scp=scp, epoch_id=epoch_id)
        tfp = es.time_from_perihelion.to_value(u.day)
        rdot = es.helio_v_kms

        if filter_type == SwiftFilter.uw1:
            print(
                f" {epoch_num} & {unique_days_str} & {rh:3.2f} & {rdot:3.1f} & {delta:3.2f} & {sto:4.2f} & {tfp:4.1f} & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f}\\\\"
            )
        else:
            print(
                f" & & & & & & & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f}\\\\"
            )


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

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    epoch_id_selected = epoch_menu(scp=scp)
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
