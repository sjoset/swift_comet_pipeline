from dataclasses import dataclass

import numpy as np


@dataclass
class ColumnDensity:
    rs_km: np.ndarray
    cd_cm2: np.ndarray
