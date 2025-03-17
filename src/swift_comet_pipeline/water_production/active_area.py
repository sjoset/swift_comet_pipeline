import astropy.units as u

from comet_ice_sublimation.model_input import SublimationModelInput
from comet_ice_sublimation.model_output import SublimationModelResult
from comet_ice_sublimation.molecular_species import MolecularSpecies
from comet_ice_sublimation.model_runner import run_sublimation_model


def make_sublimation_model_input(
    rh_au: float, sub_solar_latitude: float
) -> SublimationModelInput:

    return SublimationModelInput(
        species=MolecularSpecies.h2o,
        # TODO: cite sources for value
        visual_albedo=0.04,
        # TODO: cite sources for value
        infrared_albedo=0.05,
        rh_au=abs(rh_au),
        sub_solar_latitude=sub_solar_latitude,
        num_latitude_gridpoints=1001,
        t_init_K=180,
    )


def estimate_active_area(
    q: u.Quantity, rh: u.Quantity, sub_solar_latitude: u.Quantity
) -> u.Quantity:

    smi = make_sublimation_model_input(
        rh_au=rh.to_value(u.AU),  # type: ignore
        sub_solar_latitude=sub_solar_latitude.to_value(u.degree),  # type: ignore
    )

    smr: SublimationModelResult = run_sublimation_model(smi=smi)

    # output z_bar is in mol/cm^2/sec
    return q / (smr.z_bar / (u.cm**2 * u.s))  # type: ignore
