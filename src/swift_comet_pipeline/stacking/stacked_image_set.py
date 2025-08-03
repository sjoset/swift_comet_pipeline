from itertools import product

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.types.stacked_uvot_image_set import StackedUVOTImageSet
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter


def get_stacked_image_set(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> StackedUVOTImageSet | None:
    stacked_image_set = {}

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    for f, s in product(uw1_and_uvv, sum_and_median):
        img_data = scp.get_product_data(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        if img_data is None:
            return None
        # img_data includes img_data.header for the FITS header, and img_data.data for the numpy image array
        if img_data.data is None:
            return None
        stacked_image_set[f, s] = img_data.data

    return stacked_image_set
