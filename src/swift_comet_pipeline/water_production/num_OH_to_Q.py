from functools import cache

import astropy.units as u

from swift_comet_pipeline.modeling.vectorial_model import (
    num_OH_from_vectorial_model_result,
    num_OH_from_vectorial_model_result_within_r,
    water_vectorial_model,
)
from swift_comet_pipeline.types.hydroxyl_molecule_count import HydroxylMoleculeCount
from swift_comet_pipeline.types.water_molecule_count import WaterMoleculeCount


@cache
def num_OH_to_Q_vectorial(
    helio_r_au: float, num_OH: HydroxylMoleculeCount
) -> WaterMoleculeCount:
    base_q = 1.0e29 / u.s  # type: ignore
    helio_r = helio_r_au * u.AU  # type: ignore

    vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
    predicted_num_OH = num_OH_from_vectorial_model_result(vmr=vmr)
    predicted_to_actual = predicted_num_OH / num_OH.value

    q = base_q.value / predicted_to_actual
    q_err = (base_q.value / predicted_num_OH) * num_OH.sigma

    return WaterMoleculeCount(value=q, sigma=q_err)


@u.quantity_input
@cache
def _num_OH_within_r_to_Q_vectorial_no_err(
    helio_r_au: float, num_OH: float, within_r: u.Quantity[u.m]  # type: ignore
) -> float:
    base_q = 1.0e28 / u.s  # type: ignore
    helio_r = helio_r_au * u.AU  # type: ignore

    vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
    predicted_num_OH = num_OH_from_vectorial_model_result_within_r(
        vmr=vmr, within_r=within_r
    )
    predicted_to_actual = predicted_num_OH / num_OH

    q = base_q.value / predicted_to_actual

    return q


@u.quantity_input
@cache
def num_OH_within_r_to_Q_vectorial(
    helio_r_au: float, num_OH: HydroxylMoleculeCount, within_r: u.Quantity[u.m]  # type: ignore
) -> WaterMoleculeCount:

    lower_OH = num_OH.value - num_OH.sigma
    upper_OH = num_OH.value + num_OH.sigma

    lower_Q = _num_OH_within_r_to_Q_vectorial_no_err(
        helio_r_au=helio_r_au, num_OH=lower_OH, within_r=within_r
    )
    upper_Q = _num_OH_within_r_to_Q_vectorial_no_err(
        helio_r_au=helio_r_au, num_OH=upper_OH, within_r=within_r
    )
    q = _num_OH_within_r_to_Q_vectorial_no_err(
        helio_r_au=helio_r_au, num_OH=num_OH.value, within_r=within_r
    )

    sig_lower = abs(q - lower_Q)
    sig_upper = abs(upper_Q - q)

    return WaterMoleculeCount(value=q, sigma=(sig_lower + sig_upper) / 2)


# @u.quantity_input
# @cache
# def num_OH_within_r_to_Q_vectorial(
#     helio_r_au: float, num_OH: HydroxylMoleculeCount, within_r: u.Quantity[u.m]  # type: ignore
# ) -> WaterMoleculeCount:
#     base_q = 1.0e29 / u.s  # type: ignore
#     helio_r = helio_r_au * u.AU  # type: ignore
#
#     vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
#     predicted_num_OH = num_OH_from_vectorial_model_result_within_r(
#         vmr=vmr, within_r=within_r
#     )
#     predicted_to_actual = predicted_num_OH / num_OH.value
#
#     q = base_q.value / predicted_to_actual
#     q_err = (base_q.value / predicted_num_OH) * num_OH.sigma
#
#     return WaterMoleculeCount(value=q, sigma=q_err)
