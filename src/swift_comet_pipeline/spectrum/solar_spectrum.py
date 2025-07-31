from functools import cache

import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
from sbpy.calib import Sun

from swift_comet_pipeline.types.solar_spectrum import SolarSpectrum


@cache
def get_solar_spectrum() -> SolarSpectrum:
    sun = Sun.from_default()

    solar_lambdas_nm = sun.wave.to_value(u.nm)  # type: ignore
    spectral_irradiances_Wm2_nm = sun.fluxd.to_value(u.Watt / (u.m**2 * u.nm))  # type: ignore

    return SolarSpectrum(
        lambdas_nm=solar_lambdas_nm,
        spectral_irradiances_Wm2_nm=spectral_irradiances_Wm2_nm,
    )


def interpolate_solar_spectrum_onto(
    solar_spectrum: SolarSpectrum, lambdas_nm: np.ndarray
) -> SolarSpectrum:

    solar_irradiances_interpolation = interp1d(
        solar_spectrum.lambdas_nm, solar_spectrum.spectral_irradiances_Wm2_nm
    )
    solar_irradiances_on_lambdas = solar_irradiances_interpolation(lambdas_nm)

    return SolarSpectrum(
        lambdas_nm=lambdas_nm.copy(),
        spectral_irradiances_Wm2_nm=solar_irradiances_on_lambdas,
    )
