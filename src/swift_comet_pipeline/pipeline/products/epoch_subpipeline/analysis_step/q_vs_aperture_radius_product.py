import pathlib

from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.epoch_subpipeline_analysis_product import (
    EpochSubPipelineAnalysisProduct,
)
from swift_comet_pipeline.pipeline.products.product_io_types.ecsv_product import (
    ECSVDataframePipelineProductIO,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


class QvsApertureRadiusProduct(
    EpochSubPipelineAnalysisProduct, ECSVDataframePipelineProductIO
):

    def __init__(
        self,
        parent_epoch: EpochProduct,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        qvsa_filename = f"qh2o_vs_aperture_radius_{stacking_method}.ecsv"
        self.product_path = self.product_path / pathlib.Path(qvsa_filename)
