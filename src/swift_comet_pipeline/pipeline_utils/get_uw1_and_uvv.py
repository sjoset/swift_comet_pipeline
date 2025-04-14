from functools import cache
from swift_comet_pipeline.comet.extract_comet_radial_profile import (
    radial_profile_from_dataframe_product,
)
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.types.comet_profile import CometRadialProfile
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


@cache
def get_uw1_and_uvv_stacked_images(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> tuple[SwiftUVOTImage, SwiftUVOTImage] | None:

    stacked_uw1 = scp.get_product_data(
        pf=PipelineFilesEnum.stacked_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    stacked_uvv = scp.get_product_data(
        pf=PipelineFilesEnum.stacked_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if stacked_uw1 is None or stacked_uvv is None:
        print("Stacked images not found! Skipping")
        return None

    stacked_uw1 = stacked_uw1.data
    stacked_uvv = stacked_uvv.data

    return stacked_uw1, stacked_uvv


@cache
def get_uw1_and_uvv_extracted_radial_profiles(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> tuple[CometRadialProfile, CometRadialProfile] | None:

    uw1_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    if uw1_profile is None:
        return None

    uvv_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uvv_profile is None:
        return None

    uw1_profile = radial_profile_from_dataframe_product(df=uw1_profile)
    uvv_profile = radial_profile_from_dataframe_product(df=uvv_profile)

    return uw1_profile, uvv_profile
