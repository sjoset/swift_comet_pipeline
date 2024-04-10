import numpy as np
from functools import cache

# import pandas as pd
import sbpy.activity as sba
from sbpy.data import Phys
import astropy.units as u


# TODO: make a function to return a vmc for water at 1 AU


# @cache
# def num_OH_at_r_au_vectorial(
#     base_q_per_s: float, helio_r_au: float
# ) -> tuple[float, sba.VectorialModel]:
#     helio_r_au_sq = helio_r_au**2
#
#     # TODO: cite sources for lifetimes
#     parent_dict = {
#         "tau_d": 86000 * helio_r_au_sq * u.s,
#         "v_outflow": (0.85 / np.sqrt(helio_r_au)) * u.km / u.s,
#         "tau_T": 86000 * 0.93 * helio_r_au_sq * u.s,
#         "sigma": 3.0e-16 * (u.cm**2),
#     }
#     # TODO: check fragment tau_T
#     fragment_dict = {"v_photo": 1.05 * u.km / u.s, "tau_T": 110000 * helio_r_au_sq * u.s}  # type: ignore
#     # TODO: use pyvectorial to transform these instead of doing it manually with helio_r_au
#
#     parent = Phys.from_dict(parent_dict)  # type: ignore
#     fragment = Phys.from_dict(fragment_dict)  # type: ignore
#
#     coma = sba.VectorialModel(
#         base_q=base_q_per_s / u.s,
#         q_t=None,
#         parent=parent,
#         fragment=fragment,
#         radial_points=150,
#         angular_points=100,
#         radial_substeps=80,
#         print_progress=True,
#     )
#
#     return (coma.vmr.num_fragments_grid, coma)
