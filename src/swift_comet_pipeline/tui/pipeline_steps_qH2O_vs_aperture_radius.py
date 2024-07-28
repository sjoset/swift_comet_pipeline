import numpy as np

from swift_comet_pipeline.aperture.q_vs_aperture_radius import (
    q_vs_aperture_radius_at_epoch,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig


# TODO: show stacked images with aperture radii shaded in given the plateaus it finds
def qH2O_vs_aperture_radius_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available!")
        return

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(start=0.0, stop=50.0, num=51, endpoint=True)
    ]

    for parent_epoch in data_ingestion_files.epochs:
        esf = pipeline_files.epoch_subpipeline_from_parent_epoch(
            parent_epoch=parent_epoch
        )
        if esf is None:
            print(f"No subpipeline found for epoch {parent_epoch.product_path.name}")
            continue
        q_vs_aperture_radius_at_epoch(
            epoch_subpipeline_files=esf, dust_rednesses=dust_rednesses
        )
        print("")
