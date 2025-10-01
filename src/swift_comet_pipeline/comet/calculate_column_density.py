import numpy as np
import astropy.units as u

from swift_comet_pipeline.comet.countrate_profile_to_surface_brightness import (
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.comet.subtract_comet_profiles import subtract_profiles
from swift_comet_pipeline.fluorescence.hydroxyl_gfactor import hydroxyl_gfactor_1au
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.types import (
    ColumnDensity,
    CometCountRateProfile,
    CometRadialProfile,
    CometSurfaceBrightnessProfile,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.water_production.flux_OH import OH_count_rates_to_flux_factor


# TODO: add decorators to enforce the arguments are the correct Quantity
# TODO: return type is incorrect
def surface_brightness_profile_to_column_density(
    surface_brightness_profile: CometSurfaceBrightnessProfile,
    epoch_summary: EpochSummary,
) -> np.ndarray:

    delta_cm = (epoch_summary.delta_au * u.AU).to_value(u.cm)  # type: ignore
    alpha = OH_count_rates_to_flux_factor().to_value(u.erg / (u.cm**2 * u.s))  # type: ignore
    flux = surface_brightness_profile * alpha
    lumi = flux * 4 * np.pi * delta_cm**2

    gfactor_scaled = (
        hydroxyl_gfactor_1au(helio_v_kms=epoch_summary.helio_v_kms)
        / epoch_summary.rh_au**2
    )
    column_density = lumi / gfactor_scaled

    return column_density / (u.cm**2)  # type: ignore


def calculate_comet_column_density(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    dust_redness: DustReddeningPercent,
    r_min: u.Quantity = 1 * u.km,  # type: ignore
) -> ColumnDensity:
    # TODO: document

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert epoch_summary is not None

    subtracted_profile = subtract_profiles(
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
    )

    subtracted_profile_rs_km = (
        subtracted_profile.profile_axis_xs * epoch_summary.km_per_pix
    )

    profile_mask = subtracted_profile_rs_km > r_min.to(u.km).value  # type: ignore

    profile_rs_km = subtracted_profile_rs_km[profile_mask]
    countrate_profile: CometCountRateProfile = subtracted_profile.pixel_values[
        profile_mask
    ]

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        countrate_profile=countrate_profile,
        epoch_summary=epoch_summary,
    )

    comet_column_density_values = surface_brightness_profile_to_column_density(
        surface_brightness_profile=surface_brightness_profile,
        epoch_summary=epoch_summary,
    )

    comet_column_density = ColumnDensity(
        rs_km=profile_rs_km, cd_cm2=comet_column_density_values.to(1 / u.cm**2).value  # type: ignore
    )

    return comet_column_density
