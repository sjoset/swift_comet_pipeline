from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


class EpochSubPipelineProduct(PipelineProduct):
    """
    Base class for products in the sub-pipeline that we run for each epoch that we pull from the observation log
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_epoch = parent_epoch
        self.filter_type = filter_type
        self.stacking_method = stacking_method
