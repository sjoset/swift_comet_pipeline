import pathlib

from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.epoch_subpipeline_analysis_product import (
    EpochSubPipelineAnalysisProduct,
)
from swift_comet_pipeline.pipeline.products.product_io_types.fits_product import (
    FitsImageProductIO,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string


class MedianDividedImage(EpochSubPipelineAnalysisProduct, FitsImageProductIO):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
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

        filter_string = filter_to_file_string(filter_type=filter_type)
        mdi_filename = f"median_divided_image_{filter_string}_{stacking_method}.fits"
        self.product_path = self.product_path / pathlib.Path(mdi_filename)
