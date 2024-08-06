from functools import cache
from typing import TypeAlias

import astropy.units as u

from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
from swift_comet_pipeline.modeling.vectorial_model import (
    num_OH_from_vectorial_model_result,
    water_vectorial_model,
)
from swift_comet_pipeline.water_production.fluorescence_OH import NumOH


NumQH2O: TypeAlias = ValueAndStandardDev


@cache
def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: NumOH) -> NumQH2O:
    base_q = 1.0e29 / u.s  # type: ignore
    helio_r = helio_r_au * u.AU  # type: ignore

    vmr = water_vectorial_model(base_q=base_q, helio_r=helio_r)
    predicted_num_OH = num_OH_from_vectorial_model_result(vmr=vmr)
    predicted_to_actual = predicted_num_OH / num_OH.value

    q = base_q.value / predicted_to_actual
    q_err = (base_q.value / predicted_num_OH) * num_OH.sigma

    return NumQH2O(value=q, sigma=q_err)
