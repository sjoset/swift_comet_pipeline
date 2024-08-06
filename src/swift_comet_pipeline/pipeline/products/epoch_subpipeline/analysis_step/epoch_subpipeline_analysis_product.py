from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.epoch_subpipeline_product import (
    EpochSubPipelineProduct,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


class EpochSubPipelineAnalysisProduct(EpochSubPipelineProduct):
    """
    base class for epoch sub-pipeline analysis products - children should call super().__init()__ and the modify self.product_path to append their file name
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )
        self.product_path = (
            self.product_path / "analysis" / parent_epoch.product_path.stem
        )
        self.product_path.mkdir(parents=True, exist_ok=True)
