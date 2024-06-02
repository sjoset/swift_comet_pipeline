from dataclasses import dataclass

import numpy as np
import astropy.units as u

from swift_comet_pipeline.comet.comet_surface_brightness_profile import (
    CometSurfaceBrightnessProfile,
)
from swift_comet_pipeline.water_production.fluorescence_OH import gfactor_1au


@dataclass
class ColumnDensity:
    rs_km: np.ndarray
    cd_cm2: np.ndarray


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
