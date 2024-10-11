import pathlib
from rich import print as rprint

from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps import (
    SwiftCometPipelineStepStatus,
)
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.tui.tui_common import get_yes_no
from swift_comet_pipeline.observationlog.build_observation_log import (
    build_observation_log,
)


def observation_log_step(swift_project_config: SwiftProjectConfig) -> None:

    horizons_id = swift_project_config.jpl_horizons_id
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))
    # data_ingestion_files = DataIngestionFiles(
    #     project_path=swift_project_config.project_path
    # )

    if (
        scp.get_status(SwiftCometPipelineStepEnum.observation_log)
        == SwiftCometPipelineStepStatus.complete
    ):
        print("Observation log appears to exist - generate again and overwrite? (y/n)")
        generate_anyway = get_yes_no()
        if not generate_anyway:
            return

    # if data_ingestion_files.observation_log.exists():
    #     print("Observation log appears to exist - generate again and overwrite? (y/n)")
    #     generate_anyway = get_yes_no()
    #     if not generate_anyway:
    #         return

    print("Generating observation log ...")

    df = build_observation_log(
        swift_data=sdd,
        horizons_id=horizons_id,
    )

    if df is None:
        rprint(
            "[red]Could not construct the observation log in memory! No files written.[/red]"
        )
        return

    obs_log_prod = scp.get_product(pf=PipelineFilesEnum.observation_log)
    assert obs_log_prod is not None
    rprint(f"[green]Writing observation log to {obs_log_prod.product_path}...[/green]")
    obs_log_prod.data = df
    obs_log_prod.write()
    # data_ingestion_files.observation_log._data = df
    # data_ingestion_files.observation_log.write()
