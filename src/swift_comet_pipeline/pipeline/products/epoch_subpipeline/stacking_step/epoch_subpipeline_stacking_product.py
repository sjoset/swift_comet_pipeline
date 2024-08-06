from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.epoch_subpipeline_product import (
    EpochSubPipelineProduct,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


class EpochSubPipelineStackingProduct(EpochSubPipelineProduct):
    """
    Base class for products in the stacking pipeline step - children only need to call super().__init__() and then modify the product_path to add their filename
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
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
            self.product_path / "stacked" / parent_epoch.product_path.stem
        )
        self.product_path.mkdir(parents=True, exist_ok=True)
