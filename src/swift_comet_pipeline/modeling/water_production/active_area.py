from functools import cache
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
        # visual_albedo=0.04,
        visual_albedo=0.05,
        # TODO: cite sources for value
        # infrared_albedo=0.05,
        infrared_albedo=0.0,
        rh_au=abs(rh_au),
        sub_solar_latitude=sub_solar_latitude,
        num_latitude_gridpoints=1001,
        t_init_K=150,
    )


@cache
def estimate_active_area(
    q: u.Quantity, rh: u.Quantity, sub_solar_latitude: u.Quantity
) -> u.Quantity:

    aa_km2 = (
        estimate_active_area_km2(
            q_per_s=q.to_value(1 / u.s),  # type: ignore
            rh_au=rh.to_value(u.AU),  # type: ignore
            sub_solar_latitude_deg=sub_solar_latitude.to_value(u.deg),  # type: ignore
        )
        * u.km**2  # type: ignore
    )

    return aa_km2


# TODO: remove old code
# @cache
# def estimate_active_area(
#     q: u.Quantity, rh: u.Quantity, sub_solar_latitude: u.Quantity
# ) -> u.Quantity:
#     """
#     Runs sublimation model based on Cowan & A'Hearn 1979
#     """
#
#     if q < 0.0 / u.s:
#         return 0.0 * u.cm**2
#
#     smi = make_sublimation_model_input(
#         rh_au=rh.to_value(u.AU),  # type: ignore
#         sub_solar_latitude=sub_solar_latitude.to_value(u.degree),  # type: ignore
#     )
#
#     smr: SublimationModelResult = run_sublimation_model(smi=smi)
#
#     # output z_bar is in mol/cm^2/sec
#     return q / (smr.z_bar / (u.cm**2 * u.s))  # type: ignore


def estimate_active_area_km2(
    q_per_s: float, rh_au: float, sub_solar_latitude_deg: float
) -> float:
    """
    Runs sublimation model based on Cowan & A'Hearn 1979 to compute active area of comet surface given a production rate, etc.
    """

    if q_per_s < 0.0:
        return 0.0

    smi = make_sublimation_model_input(
        rh_au=rh_au, sub_solar_latitude=sub_solar_latitude_deg
    )

    smr: SublimationModelResult = run_sublimation_model(smi=smi)

    # output z_bar is in mol/cm^2/sec
    z_bar_km2 = smr.z_bar * 1e10

    return q_per_s / z_bar_km2
