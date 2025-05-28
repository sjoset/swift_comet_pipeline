from functools import cache

from swift_comet_pipeline.comet.extract_comet_radial_profile import (
    radial_profile_from_dataframe_product,
)
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.types.background_result import yaml_dict_to_background_result
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.uw1_uvv_pair import Uw1UvvPair


@cache
def get_uw1_and_uvv_stacked_images(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> Uw1UvvPair | None:

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

    stacked_uw1_img = stacked_uw1.data if stacked_uw1 is not None else None
    stacked_uvv_img = stacked_uvv.data if stacked_uvv is not None else None

    if stacked_uw1_img is None or stacked_uvv_img is None:
        return None

    return {SwiftFilter.uw1: stacked_uw1_img, SwiftFilter.uvv: stacked_uvv_img}


@cache
def get_uw1_and_uvv_median_subtracted_images(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> Uw1UvvPair | None:

    sub_uw1 = scp.get_product_data(
        pf=PipelineFilesEnum.median_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    sub_uvv = scp.get_product_data(
        pf=PipelineFilesEnum.median_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )

    sub_uw1_img = sub_uw1.data if sub_uw1 is not None else None
    sub_uvv_img = sub_uvv.data if sub_uvv is not None else None

    if sub_uw1_img is None or sub_uvv_img is None:
        return None

    return {SwiftFilter.uw1: sub_uw1_img, SwiftFilter.uvv: sub_uvv_img}


@cache
def get_uw1_and_uvv_background_subtracted_images(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> Uw1UvvPair | None:

    bg_sub_uw1 = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    bg_sub_uvv = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )

    bg_sub_uw1_img = bg_sub_uw1.data if bg_sub_uw1 is not None else None
    bg_sub_uvv_img = bg_sub_uvv.data if bg_sub_uvv is not None else None

    if bg_sub_uw1_img is None or bg_sub_uvv_img is None:
        return None

    return {SwiftFilter.uw1: bg_sub_uw1_img, SwiftFilter.uvv: bg_sub_uvv_img}


@cache
def get_uw1_and_uvv_extracted_radial_profiles(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> Uw1UvvPair | None:

    uw1_profile_df = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv_profile_df = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )

    uw1_profile = (
        radial_profile_from_dataframe_product(df=uw1_profile_df)
        if uw1_profile_df is not None
        else None
    )
    uvv_profile = (
        radial_profile_from_dataframe_product(df=uvv_profile_df)
        if uvv_profile_df is not None
        else None
    )

    if uw1_profile is None or uvv_profile is None:
        return None

    return {SwiftFilter.uw1: uw1_profile, SwiftFilter.uvv: uvv_profile}


@cache
def get_uw1_and_uvv_background_results(
    scp: SwiftCometPipeline, epoch_id: EpochID, stacking_method: StackingMethod
) -> Uw1UvvPair | None:

    uw1_bg_yaml = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv_bg_yaml = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )

    uw1_bg = (
        yaml_dict_to_background_result(raw_yaml=uw1_bg_yaml)
        if uw1_bg_yaml is not None
        else None
    )
    uvv_bg = (
        yaml_dict_to_background_result(raw_yaml=uvv_bg_yaml)
        if uvv_bg_yaml is not None
        else None
    )

    if uw1_bg is None or uvv_bg is None:
        return None

    return {SwiftFilter.uw1: uw1_bg, SwiftFilter.uvv: uvv_bg}
