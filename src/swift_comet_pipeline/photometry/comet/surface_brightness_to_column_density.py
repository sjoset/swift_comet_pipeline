import numpy as np
import astropy.units as u

from swift_comet_pipeline.modeling.fluorescence.hydroxyl_gfactor import (
    hydroxyl_gfactor_1au,
)
from swift_comet_pipeline.modeling.water_production.flux_OH import (
    oh_count_rates_to_flux_factor,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


def surface_brightness_profile_to_oh_column_density(
    eid: EpochIndexEntry,
    surface_brightness_profile: CometSurfaceBrightnessProfile,
    oh_filter: UvotFilter,
) -> np.ndarray:
    # return type has Quantity attached - num oh per cm^2

    delta_cm: float = (eid.delta_au * u.AU).to_value(u.cm)  # type: ignore
    alpha = oh_count_rates_to_flux_factor(filter_type=oh_filter).to_value(
        u.erg / (u.cm**2 * u.s)  # type: ignore
    )
    flux = surface_brightness_profile * alpha
    lumi = flux * 4 * np.pi * delta_cm**2

    gfactor_scaled = hydroxyl_gfactor_1au(helio_v_kms=eid.helio_v_kms) / eid.rh_au**2
    column_density = lumi / gfactor_scaled

    return column_density / (u.cm**2)
