from functools import cache

import astropy.units as u
import numpy as np

from swift_comet_pipeline.builders.vectorial_model_cacher import (
    get_vectorial_model_cache_Q,
)
from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    num_oh_from_vectorial_model_result,
    num_oh_from_vectorial_model_result_within_r,
    water_vectorial_model,
)
from swift_comet_pipeline.scp_types.compound.hydroxyl_molecule_count import (
    HydroxylMoleculeCount,
)
from swift_comet_pipeline.scp_types.primitive.error_propogation import (
    ValueAndStandardDev,
)


@cache
def num_oh_to_q_h2o_vectorial(
    rh_au: float, num_oh: HydroxylMoleculeCount
) -> ValueAndStandardDev:
    """
    Using a vectorial model, takes the number of oh molecules and returns
    the production rate that produces the same *total* number of oh molecules
    """
    base_q = get_vectorial_model_cache_Q()
    helio_r = rh_au * u.AU  # type: ignore

    vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
    predicted_num_oh = num_oh_from_vectorial_model_result(vmr=vmr)
    predicted_to_actual = predicted_num_oh / num_oh.value

    q = base_q.value / predicted_to_actual
    q_err = (base_q.value / predicted_num_oh) * num_oh.sigma

    return ValueAndStandardDev(value=q, sigma=q_err)


@u.quantity_input
@cache
def _num_oh_within_r_to_q_h2o_vectorial_no_err(
    rh_au: float, num_oh: float, within_r: u.Quantity[u.m]  # type: ignore
) -> float:
    base_q = get_vectorial_model_cache_Q()
    helio_r = rh_au * u.AU  # type: ignore

    vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
    predicted_num_oh = num_oh_from_vectorial_model_result_within_r(
        vmr=vmr, within_r=within_r
    )
    predicted_to_actual = predicted_num_oh / num_oh

    q = base_q.value / predicted_to_actual

    return q


@u.quantity_input
@cache
def num_oh_within_r_to_q_h2o_vectorial(
    rh_au: float, num_oh: HydroxylMoleculeCount, within_r: u.Quantity[u.m]  # type: ignore
) -> ValueAndStandardDev:

    lower_oh = num_oh.value - num_oh.sigma
    upper_oh = num_oh.value + num_oh.sigma

    lower_Q = _num_oh_within_r_to_q_h2o_vectorial_no_err(
        rh_au=rh_au, num_oh=lower_oh, within_r=within_r
    )
    upper_Q = _num_oh_within_r_to_q_h2o_vectorial_no_err(
        rh_au=rh_au, num_oh=upper_oh, within_r=within_r
    )
    q = _num_oh_within_r_to_q_h2o_vectorial_no_err(
        rh_au=rh_au, num_oh=num_oh.value, within_r=within_r
    )

    var_lower = (q - lower_Q) ** 2
    var_upper = (upper_Q - q) ** 2
    total_sig = np.sqrt(var_lower + var_upper)

    return ValueAndStandardDev(value=q, sigma=total_sig)
