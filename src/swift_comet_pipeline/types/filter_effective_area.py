from dataclasses import dataclass

import numpy as np


@dataclass
class FilterEffectiveArea:
    lambdas_nm: np.ndarray
    responses_cm2: np.ndarray
