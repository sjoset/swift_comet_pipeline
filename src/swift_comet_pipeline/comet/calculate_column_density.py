import numpy as np
import astropy.units as u

from swift_comet_pipeline.comet.column_density import ColumnDensity
from swift_comet_pipeline.comet.comet_count_rate_profile import CometCountRateProfile
from swift_comet_pipeline.comet.comet_radial_profile import (
    CometRadialProfile,
    subtract_profiles,
)
from swift_comet_pipeline.comet.comet_surface_brightness_profile import (
    CometSurfaceBrightnessProfile,
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.stacked_epoch import StackedEpoch
from swift_comet_pipeline.swift.uvot_image import datamode_to_pixel_resolution
from swift_comet_pipeline.water_production.fluorescence_OH import gfactor_1au


# TODO: add decorators to enforce the arguments are the correct Quantity
def surface_brightness_profile_to_column_density(
    surface_brightness_profile: CometSurfaceBrightnessProfile,
    delta: u.Quantity,
    helio_v: u.Quantity,
    helio_r: u.Quantity,
) -> np.ndarray:
    delta_cm = delta.to(u.cm).value  # type: ignore
    helio_v_kms = helio_v.to(u.km / u.s).value  # type: ignore
    rh_au = helio_r.to(u.AU).value  # type: ignore

    # TODO: magic numbers
    # specific to OH - this should be from Bodewits but ask Lucy to make sure
    alpha = 1.2750906353215913e-12
    flux = surface_brightness_profile * alpha
    lumi = flux * 4 * np.pi * delta_cm**2

    gfactor = gfactor_1au(helio_v_kms=helio_v_kms) / rh_au**2
    column_density = lumi / gfactor

    return column_density / (u.cm**2)  # type: ignore


def calculate_comet_column_density(
    stacked_epoch: StackedEpoch,
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    dust_redness: DustReddeningPercent,
    r_min: u.Quantity = 1 * u.km,  # type: ignore
) -> ColumnDensity:
    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    helio_v = np.mean(stacked_epoch.HELIO_V) * (u.km / u.s)  # type: ignore
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    pixel_resolution = datamode_to_pixel_resolution(stacked_epoch.DATAMODE[0])

    subtracted_profile = subtract_profiles(
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
    )

    subtracted_profile_rs_km = subtracted_profile.profile_axis_xs * km_per_pix

    profile_mask = subtracted_profile_rs_km > r_min.to(u.km).value  # type: ignore

    profile_rs_km = subtracted_profile_rs_km[profile_mask]
    countrate_profile: CometCountRateProfile = subtracted_profile.pixel_values[
        profile_mask
    ]

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        countrate_profile=countrate_profile,
        pixel_resolution=pixel_resolution,
        delta=delta,
    )

    comet_column_density_values = surface_brightness_profile_to_column_density(
        surface_brightness_profile=surface_brightness_profile,
        delta=delta,
        helio_v=helio_v,
        helio_r=helio_r,
    )

    comet_column_density = ColumnDensity(
        rs_km=profile_rs_km, cd_cm2=comet_column_density_values.to(1 / u.cm**2).value  # type: ignore
    )

    return comet_column_density
