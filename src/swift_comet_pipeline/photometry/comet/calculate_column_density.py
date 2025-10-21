import astropy.units as u

from swift_comet_pipeline.photometry.comet.countrate_profile_to_surface_brightness import (
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.photometry.comet.subtract_comet_profiles import (
    subtract_profiles,
)
from swift_comet_pipeline.photometry.comet.surface_brightness_to_column_density import (
    surface_brightness_profile_to_oh_column_density,
)
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


def calculate_oh_column_density(
    eid: EpochIndexEntry,
    oh_profile: CometRadialProfileFromConicalRegion,
    dust_profile: CometRadialProfileFromConicalRegion,
    dust_redness: DustReddeningPercent,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    r_min: u.Quantity = 1 * u.km,  # type: ignore
) -> ColumnDensity:
    # TODO: document

    subtracted_profile = subtract_profiles(
        oh_profile=oh_profile,
        dust_profile=dust_profile,
        dust_redness=dust_redness,
        oh_filter=oh_filter,
        dust_filter=dust_filter,
    )

    subtracted_profile_rs_km = subtracted_profile.profile_axis_rs * eid.km_per_pix

    profile_mask = subtracted_profile_rs_km > r_min.to(u.km).value  # type: ignore

    profile_rs_km = subtracted_profile_rs_km[profile_mask]
    countrate_profile: CometCountRateProfile = subtracted_profile.pixel_values[
        profile_mask
    ]

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        eid=eid, countrate_profile=countrate_profile
    )

    comet_column_density_values = surface_brightness_profile_to_oh_column_density(
        eid=eid,
        surface_brightness_profile=surface_brightness_profile,
        oh_filter=oh_filter,
    )

    comet_column_density = ColumnDensity(
        rs_km=profile_rs_km, cd_cm2=comet_column_density_values.to(1 / u.cm**2).value  # type: ignore
    )

    return comet_column_density
