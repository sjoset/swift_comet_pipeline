from functools import cache
import astropy.units as u

from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.get_uw1_and_uvv import (
    get_uw1_and_uvv_extracted_radial_profiles,
)
from swift_comet_pipeline.types.column_density import ColumnDensity
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.uw1_uvv_pair import uw1uvv_getter


@cache
def get_comet_column_density_from_extracted_profile(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    dust_redness: DustReddeningPercent,
    stacking_method: StackingMethod,
) -> ColumnDensity:

    radial_profile_pair = get_uw1_and_uvv_extracted_radial_profiles(
        scp=scp, epoch_id=epoch_id, stacking_method=stacking_method
    )
    assert radial_profile_pair is not None
    uw1_profile, uvv_profile = uw1uvv_getter(radial_profile_pair)

    comet_cd = calculate_comet_column_density(
        scp=scp,
        epoch_id=epoch_id,
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
        r_min=1 * u.km,  # type: ignore
    )

    return comet_cd
