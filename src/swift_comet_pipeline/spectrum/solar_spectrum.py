from functools import cache

import astropy.units as u
from sbpy.calib import Sun

from swift_comet_pipeline.types.solar_spectrum import SolarSpectrum


# @cache
# def read_fixed_solar_spectrum(solar_spectrum_path: pathlib.Path) -> SolarSpectrum:
#     """
#     Reads a modeled solar spectrum csv file of the form (lambda, irradiance) and returns a SolarSpectrum
#     Colina et al 1996
#     """
#     # load the solar spectrum
#     solar_spectrum_df = pd.read_csv(solar_spectrum_path)
#
#     # convert lambda from angstroms to nanometers
#     solar_lambdas = (
#         solar_spectrum_df["wavelength (angstroms)"].to_numpy(dtype=float) / 10
#     )
#     # factor of 100 to convert to correct units
#     solar_irradiances = solar_spectrum_df["irradiance"].to_numpy(dtype=float) / 100
#
#     return SolarSpectrum(
#         lambdas_nm=solar_lambdas, spectral_irradiances_Wm2_nm=solar_irradiances
#     )


@cache
def get_solar_spectrum() -> SolarSpectrum:
    sun = Sun.from_default()

    solar_lambdas_nm = sun.wave.to_value(u.nm)  # type: ignore
    spectral_irradiances_Wm2_nm = sun.fluxd.to_value(u.Watt / (u.m**2 * u.nm))  # type: ignore

    return SolarSpectrum(
        lambdas_nm=solar_lambdas_nm,
        spectral_irradiances_Wm2_nm=spectral_irradiances_Wm2_nm,
    )
