from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.epoch_subpipeline_stacking_product import (
    EpochSubPipelineStackingProduct,
)


class StackedEpochProduct(EpochSubPipelineStackingProduct, EpochProduct):
    """
    These epochs have all DataFrame rows removed that were not included in image stacking, like images taken in non-UW1 or non-UVV filters.
    This removes the need for logic to check if any particular dataframe row was included or not - we can read this epoch and assume every image mentioned is included.
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=None,
            stacking_method=None,
            *args,
            **kwargs,
        )

        # use the same name as the parent epoch
        self.product_path = self.product_path / parent_epoch.product_path.name
