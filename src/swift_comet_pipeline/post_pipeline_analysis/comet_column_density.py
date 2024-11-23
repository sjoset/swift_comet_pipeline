from functools import cache
import astropy.units as u

from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)
from swift_comet_pipeline.comet.column_density import ColumnDensity
from swift_comet_pipeline.comet.comet_radial_profile import (
    radial_profile_from_dataframe_product,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


@cache
def get_comet_column_density(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    dust_redness: DustReddeningPercent,
    stacking_method: StackingMethod,
) -> ColumnDensity:
    uw1_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_profile is not None
    uw1_profile = radial_profile_from_dataframe_product(df=uw1_profile)
    uvv_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_profile is not None
    uvv_profile = radial_profile_from_dataframe_product(df=uvv_profile)

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert stacked_epoch is not None

    comet_cd = calculate_comet_column_density(
        stacked_epoch=stacked_epoch,
        dust_redness=dust_redness,
        r_min=1 * u.km,  # type: ignore
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
    )

    return comet_cd
