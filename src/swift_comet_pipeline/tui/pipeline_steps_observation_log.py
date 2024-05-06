import pathlib
from rich import print as rprint

from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.tui.tui_common import get_yes_no, wait_for_key
from swift_comet_pipeline.observationlog.observation_log import build_observation_log

from swift_comet_pipeline.pipeline.pipeline_products import (
    DataIngestionFiles,
)


def observation_log_step(swift_project_config: SwiftProjectConfig) -> None:
    horizons_id = swift_project_config.jpl_horizons_id
    sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))
    data_ingestion_files = DataIngestionFiles(
        project_path=swift_project_config.project_path
    )

    if data_ingestion_files.observation_log.exists:
        print("Observation log appears to exist - generate again and overwrite? (y/n)")
        generate_anyway = get_yes_no()
        if not generate_anyway:
            return

    print("Generating observation log ...")

    df = build_observation_log(
        swift_data=sdd,
        obsids=sdd.get_all_observation_ids(),
        horizons_id=horizons_id,
    )

    if df is None:
        rprint(
            "[red]Could not construct the observation log in memory! No files written.[/red]"
        )
        return

    data_ingestion_files.observation_log._data = df
    print(
        f"Writing observation log to {data_ingestion_files.observation_log.product_path}..."
    )
    data_ingestion_files.observation_log.write()
    rprint("[green]Complete![/green]")
    wait_for_key()


# def observation_log_step(swift_project_config: SwiftProjectConfig) -> None:
#     horizons_id = swift_project_config.jpl_horizons_id
#     sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))
#     pipeline_files = PipelineFiles(swift_project_config.product_save_path)
#
#     if pipeline_files.exists(p=PipelineProductType.observation_log):
#         print("Observation log appears to exist - generate again and overwrite? (y/n)")
#         generate_anyway = get_yes_no()
#         if not generate_anyway:
#             return
#
#     print("Generating observation log ...")
#
#     df = build_observation_log(
#         swift_data=sdd,
#         obsids=sdd.get_all_observation_ids(),
#         horizons_id=horizons_id,
#     )
#
#     if df is None:
#         rprint(
#             "[red]Could not construct the observation log in memory! No files written.[/red]"
#         )
#         return
#
#     print(f"Writing observation log to {pipeline_files.observation_log_path}...")
#     pipeline_files.write_pipeline_product(
#         p=PipelineProductType.observation_log, data=df
#     )
#     rprint("[green]Complete![/green]")
#     wait_for_key()
