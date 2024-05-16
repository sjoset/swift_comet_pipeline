from typing import TypeAlias

import astropy.units as u

from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
from swift_comet_pipeline.modeling.vectorial_model import num_OH_at_r_au_vectorial
from swift_comet_pipeline.water_production.fluorescence_OH import NumOH


NumQH2O: TypeAlias = ValueAndStandardDev


def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: NumOH) -> NumQH2O:
    base_q = 1.0e29 / u.s

    predicted_num_OH, _ = num_OH_at_r_au_vectorial(
        base_q_per_s=base_q.to_value(1 / u.s), helio_r_au=helio_r_au
    )
    predicted_to_actual = predicted_num_OH / num_OH.value

    q = base_q.value / predicted_to_actual
    q_err = (base_q.value / predicted_num_OH) * num_OH.sigma

    return NumQH2O(value=q, sigma=q_err)
