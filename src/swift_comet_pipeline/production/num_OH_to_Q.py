from functools import cache
from typing import TypeAlias

import numpy as np
import sbpy.activity as sba
from sbpy.data import Phys
import astropy.units as u

from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
from swift_comet_pipeline.production.fluorescence_OH import NumOH


NumQH2O: TypeAlias = ValueAndStandardDev


# def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: NumOH) -> NumQH2O:
#     # TODO: this is incredibly to ugly to hard-code vm_Q here
#
#     # dummy water production that the models were run with
#     vm_Q = 1.0e28
#
#     spc = read_swift_pipeline_config()
#     if spc is None:
#         print("Could not load pipeline config")
#         return NumQH2O(value=0, sigma=0)
#
#     vectorial_model_path = spc.vectorial_model_path
#     vm_df = pd.read_csv(vectorial_model_path)
#
#     r_vs_num_OH_interp = interp1d(
#         vm_df["r (AU)"], vm_df["total_fragment_number"], fill_value="extrapolate"  # type: ignore
#     )
#
#     predicted_num_OH = r_vs_num_OH_interp(helio_r_au)
#
#     predicted_to_actual = predicted_num_OH / num_OH.value
#
#     q = vm_Q / predicted_to_actual
#     q_err = (vm_Q / predicted_num_OH) * num_OH.sigma
#
#     return NumQH2O(value=q, sigma=q_err)


@cache
def num_OH_at_r_au_vectorial(
    base_q_per_s: float, helio_r_au: float
) -> tuple[float, sba.VectorialModel]:
    helio_r_au_sq = helio_r_au**2

    # TODO: cite sources for lifetimes
    parent_dict = {
        "tau_d": 86000 * helio_r_au_sq * u.s,
        "v_outflow": (0.85 / np.sqrt(helio_r_au)) * u.km / u.s,
        "tau_T": 86000 * 0.93 * helio_r_au_sq * u.s,
        "sigma": 3.0e-16 * (u.cm**2),
    }
    # TODO: check fragment tau_T
    fragment_dict = {"v_photo": 1.05 * u.km / u.s, "tau_T": 110000 * helio_r_au_sq * u.s}  # type: ignore
    # TODO: use pyvectorial to transform these instead of doing it manually with helio_r_au

    parent = Phys.from_dict(parent_dict)  # type: ignore
    fragment = Phys.from_dict(fragment_dict)  # type: ignore

    coma = sba.VectorialModel(
        base_q=base_q_per_s / u.s,
        q_t=None,
        parent=parent,
        fragment=fragment,
        radial_points=150,
        angular_points=100,
        radial_substeps=80,
        print_progress=True,
    )

    return (coma.vmr.num_fragments_grid, coma)


def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: NumOH) -> NumQH2O:
    base_q = 1.0e29 / u.s

    predicted_num_OH, _ = num_OH_at_r_au_vectorial(
        base_q_per_s=base_q.to_value(1 / u.s), helio_r_au=helio_r_au
    )
    predicted_to_actual = predicted_num_OH / num_OH.value

    q = base_q.value / predicted_to_actual
    q_err = (base_q.value / predicted_num_OH) * num_OH.sigma

    return NumQH2O(value=q, sigma=q_err)
