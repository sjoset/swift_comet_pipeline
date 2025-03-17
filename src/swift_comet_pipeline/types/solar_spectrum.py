from dataclasses import dataclass

import numpy as np


# TODO: use sbpy's included solar spectra:
# https://sbpy.readthedocs.io/en/stable/sbpy/calib.html
# for Colina et al 1996 instead of having a copy of our own


@dataclass
class SolarSpectrum:
    lambdas_nm: np.ndarray  # nanometers
    spectral_irradiances_Wm2_nm: np.ndarray  # watts per meter squared per nanometer
