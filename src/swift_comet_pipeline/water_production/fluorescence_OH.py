import numpy as np
import astropy.units as u


from swift_comet_pipeline.fluorescence.hydroxyl_gfactor import hydroxyl_gfactor_1au
from swift_comet_pipeline.types.hydroxyl_molecule_count import HydroxylMoleculeCount
from swift_comet_pipeline.water_production.flux_OH import OHFlux


# TODO: change helio_v_kms, helio_r_au, delta_au to be an astropy quantity, add decorator to enforce proper input
def flux_OH_to_num_OH(
    flux_OH: OHFlux,
    helio_r_au: float,
    helio_v_kms: float,
    delta_au: float,
) -> HydroxylMoleculeCount:
    # flux_OH given in ergs/cm**2/second

    # g factors given in terms of ergs, so we need to use cm while calculating luminescence
    delta = (delta_au * u.AU).to_value(u.cm)  # type: ignore
    luminescence = 4 * np.pi * flux_OH * delta**2

    g_factor = hydroxyl_gfactor_1au(helio_v_kms=helio_v_kms) / (helio_r_au**2)

    num_OH = luminescence / g_factor

    return HydroxylMoleculeCount(value=num_OH.value, sigma=num_OH.sigma)
