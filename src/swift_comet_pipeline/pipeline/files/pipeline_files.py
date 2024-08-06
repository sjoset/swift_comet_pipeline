import pathlib
from typing import List

from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.pipeline.files.data_ingestion_files import DataIngestionFiles
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.lightcurve.lightcurve_products import (
    BestRednessLightCurveProduct,
    CompleteVectorialLightCurveProduct,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod

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

        self.epoch_subpipelines: List[EpochSubpipelineFiles] | None = None
        if self.data_ingestion_files.epochs is None:
            return

        # TODO: just make this a dict of parent EpochProduct -> EpochSubpipelineFiles
        self.epoch_subpipelines = [
            EpochSubpipelineFiles(project_path=self.project_path, parent_epoch=x)
            for x in self.data_ingestion_files.epochs
        ]

        self.complete_vectorial_lightcurves = {}
        self.best_near_fit_lightcurves = {}
        self.best_far_fit_lightcurves = {}
        self.best_full_fit_lightcurves = {}

        for stacking_method in [StackingMethod.summation, StackingMethod.median]:
            self.complete_vectorial_lightcurves[stacking_method] = (
                CompleteVectorialLightCurveProduct(
                    product_path=self.project_path, stacking_method=stacking_method
                )
            )
            self.best_near_fit_lightcurves[stacking_method] = (
                BestRednessLightCurveProduct(
                    product_path=self.project_path,
                    stacking_method=stacking_method,
                    fit_type=VectorialFitType.near_fit,
                )
            )
            self.best_far_fit_lightcurves[stacking_method] = (
                BestRednessLightCurveProduct(
                    product_path=self.project_path,
                    stacking_method=stacking_method,
                    fit_type=VectorialFitType.far_fit,
                )
            )
            self.best_full_fit_lightcurves[stacking_method] = (
                BestRednessLightCurveProduct(
                    product_path=self.project_path,
                    stacking_method=stacking_method,
                    fit_type=VectorialFitType.full_fit,
                )
            )

    def epoch_subpipeline_from_parent_epoch(
        self, parent_epoch: EpochProduct
    ) -> EpochSubpipelineFiles | None:
        if self.epoch_subpipelines is None:
            return None

        matching_subs = [
            x for x in self.epoch_subpipelines if x.parent_epoch == parent_epoch
        ]
        assert len(matching_subs) == 1

        return matching_subs[0]


# class PipelineStep(Protocol):
#     def check_dependencies(self) -> bool: ...
