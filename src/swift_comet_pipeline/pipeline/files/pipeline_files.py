import pathlib
from typing import Optional, List

from swift_comet_pipeline.pipeline.files.data_ingestion_files import DataIngestionFiles
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)

# TODO: find out why stacked epoch products are saying every entry is uw1 filter

# class PipelineProductType(StrEnum):
#     observation_log = auto()
#     comet_orbital_data = auto()
#     earth_orbital_data = auto()
#     epoch = auto()
#     stacked_epoch = auto()
#     stacked_image = auto()
#     stacked_image_header = auto()
#     background_analysis = auto()
#     background_subtracted_image = auto()
#     qh2o_vs_aperture_radius = auto()
#     qh2o_from_profile = auto()


# class DataIngestionPipelineStep(StrEnum):
#     observation_log = auto()
#     comet_orbital_data = auto()
#     earth_orbital_data = auto()
#     epoch_slicing = auto()
#     data_ingestion_complete = auto()


# class EpochProcessingPipelineStep(StrEnum):
#     image_stacking = auto()
#     background_subtraction = auto()
#     qvsr_analysis = auto()
#     q_from_profile = auto()


class PipelineFiles:
    def __init__(self, project_path: pathlib.Path):
        self.project_path = project_path
        self.data_ingestion_files = DataIngestionFiles(project_path=self.project_path)

        self.epoch_subpipelines: Optional[List[EpochSubpipelineFiles]] = None
        if self.data_ingestion_files.epochs is None:
            return

        # TODO: just make this a dict of parent EpochProduct -> EpochSubpipelineFiles
        self.epoch_subpipelines = [
            EpochSubpipelineFiles(project_path=self.project_path, parent_epoch=x)
            for x in self.data_ingestion_files.epochs
        ]

    def epoch_subpipeline_from_parent_epoch(
        self, parent_epoch: EpochProduct
    ) -> Optional[EpochSubpipelineFiles]:
        if self.epoch_subpipelines is None:
            return None

        matching_subs = [
            x for x in self.epoch_subpipelines if x.parent_epoch == parent_epoch
        ]
        assert len(matching_subs) == 1

        return matching_subs[0]


# class PipelineStep(Protocol):
#     def check_dependencies(self) -> bool: ...
