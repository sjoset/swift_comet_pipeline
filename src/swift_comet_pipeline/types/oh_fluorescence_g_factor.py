from dataclasses import dataclass

import numpy as np


# TODO: add entry to identify the molecule this gfactor is describing or rename this for hydroxyl
@dataclass(frozen=True)
class OHFluorescenceGFactor1AU:
    # in km/s
    helio_v_kms: np.ndarray
    # in erg/s/molecule
    gfactors_erg_s_molecule: np.ndarray
