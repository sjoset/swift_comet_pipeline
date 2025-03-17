from dataclasses import dataclass

import numpy as np


# TODO:
# https://asteroid.lowell.edu/comet/gfactor


# TODO: add entry to identify the molecule this gfactor is describing or rename this for hydroxyl
@dataclass(frozen=True)
class FluorescenceGFactor1AU:
    helio_vs: np.ndarray
    gfactors: np.ndarray
