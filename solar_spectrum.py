#!/usr/bin/env python3

import pathlib
import numpy as np
import pandas as pd

from astropy.time import Time

from dataclasses import dataclass


@dataclass
class SolarSpectrum:
    lambdas: np.ndarray
    irradiances: np.ndarray


__version__ = "0.0.1"

__all__ = [
    "read_solar_spectrum",
    "read_solar_spectrum_sorce",
    "get_sorce_spectrum",
    "SolarSpectrum",
]


def read_solar_spectrum(solar_spectrum_path: pathlib.Path) -> SolarSpectrum:
    """
    Reads a modeled solar spectrum csv file of the form (lambda, irradiance) and returns a SolarSpectrum
    """
    # load the solar spectrum
    solar_spectrum_df = pd.read_csv(solar_spectrum_path)

    # TODO: change these to .values()?
    # convert lambda to nanometers
    solar_lambdas = solar_spectrum_df["wavelength (angstroms)"] / 10
    solar_irradiances = solar_spectrum_df["irradiance"]

    return SolarSpectrum(lambdas=solar_lambdas, irradiances=solar_irradiances)


def read_solar_spectrum_sorce(
    solar_spectrum_path: pathlib.Path, solar_spectrum_time: Time
) -> SolarSpectrum:
    # load the solar spectrum
    solar_spectrum_df = pd.read_csv(solar_spectrum_path)

    # select the spectrum from the current date
    solar_spectrum_df["time (Julian Date)"].map(lambda x: Time(x, format="jd"))
    solar_mask = solar_spectrum_df["time (Julian Date)"] == np.round(
        solar_spectrum_time.jd
    )
    solar_spectrum = solar_spectrum_df[solar_mask]

    solar_lambdas = solar_spectrum["wavelength (nm)"]
    solar_irradiances = solar_spectrum["irradiance (W/m^2/nm)"]

    return SolarSpectrum(lambdas=solar_lambdas, irradiances=solar_irradiances)


# def read_solar_spectrum_sorce_full(solar_spectrum_path: pathlib.Path) -> SolarSpectrum:
#     # load the solar spectrum
#     solar_spectrum = pd.read_csv(solar_spectrum_path)
#
#     # select the spectrum from the current date
#     solar_spectrum["time (Julian Date)"].map(lambda x: Time(x, format="jd"))
#
#     # the sorce spectra include some non-contiguous information at low wavelengths
#     mask = solar_spectrum["wavelength (nm)"] > 115.0
#
#     solar_lambdas = solar_spectrum["wavelength (nm)"][mask]
#     solar_irradiances = solar_spectrum["irradiance (W/m^2/nm)"][mask]
#
#     return SolarSpectrum(lambdas=solar_lambdas, irradiances=solar_irradiances)


def get_sorce_spectrum(t: Time) -> SolarSpectrum:
    """
    Looks in the directory data/solar_spectra/[year] for the file sorce_ssi_l3.csv, which should contain columns
    named 'time (Julian Date)', 'wavelength (nm)' and 'irradiance (W/m^2/nm)'
    """
    year = t.to_datetime().date().year
    spectrum_path = pathlib.Path(
        "data/solar_spectra/sorce/" + str(year) + "/sorce_ssi_l3.csv"
    )
    ss = read_solar_spectrum_sorce(spectrum_path, t)
    # ss.irradiances = ss.irradiances * 100
    return ss
