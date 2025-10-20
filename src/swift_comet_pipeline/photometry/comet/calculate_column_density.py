import numpy as np
import astropy.units as u

from swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor import (
    hydroxyl_gfactor_1au,
)
from swift_comet_pipeline.modeling.water_production.flux_OH import (
    oh_count_rates_to_flux_factor,
)
from swift_comet_pipeline.photometry.comet.countrate_profile_to_surface_brightness import (
    oh_countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.photometry.comet.subtract_comet_profiles import (
    subtract_profiles,
)
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


# TODO: add decorators to enforce the arguments are the correct Quantity
def surface_brightness_profile_to_oh_column_density(
    eid: EpochIndexEntry,
    surface_brightness_profile: CometSurfaceBrightnessProfile,
    oh_filter: UvotFilter,
) -> np.ndarray:
    # return type has Quantity attached - num oh per cm^2

    delta_cm = (eid.delta_au * u.AU).to_value(u.cm)  # type: ignore
    alpha = oh_count_rates_to_flux_factor(filter_type=oh_filter).to_value(
        u.erg / (u.cm**2 * u.s)  # type: ignore
    )
    flux = surface_brightness_profile * alpha
    lumi = flux * 4 * np.pi * delta_cm**2

    gfactor_scaled = hydroxyl_gfactor_1au(helio_v_kms=eid.helio_v_kms) / eid.rh_au**2
    column_density = lumi / gfactor_scaled

    return column_density / (u.cm**2)


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

    surface_brightness_profile = oh_countrate_profile_to_surface_brightness(
        eid=eid, oh_countrate_profile=countrate_profile
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
