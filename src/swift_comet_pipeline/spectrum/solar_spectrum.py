import pathlib
import numpy as np
import pandas as pd

from astropy.time import Time
from dataclasses import dataclass
from scipy.interpolate import interp1d

from swift_comet_pipeline.swift.swift_filter import read_effective_area


@dataclass
class SolarSpectrum:
    # TODO: tag with astropy units or rename to lambdas_nm, irradiances_{unit}
    lambdas: np.ndarray
    irradiances: np.ndarray


def read_fixed_solar_spectrum(solar_spectrum_path: pathlib.Path) -> SolarSpectrum:
    """
    Reads a modeled solar spectrum csv file of the form (lambda, irradiance) and returns a SolarSpectrum
    """
    # load the solar spectrum
    solar_spectrum_df = pd.read_csv(solar_spectrum_path)

    # convert lambda from angstroms to nanometers
    solar_lambdas = (
        solar_spectrum_df["wavelength (angstroms)"].to_numpy(dtype=float) / 10
    )
    solar_irradiances = solar_spectrum_df["irradiance"].to_numpy(dtype=float)

    return SolarSpectrum(lambdas=solar_lambdas, irradiances=solar_irradiances)


# TODO: test this
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

    # TODO: put the dataframe columns into an np.array() and test
    solar_lambdas = solar_spectrum["wavelength (nm)"]
    solar_irradiances = solar_spectrum["irradiance (W/m^2/nm)"]

    return SolarSpectrum(
        lambdas=np.array(solar_lambdas), irradiances=np.array(solar_irradiances)
    )


# TODO: this is probably broken: make it use the base directory of the code instead of relative path in spectrum_path
def get_sorce_spectrum(t: Time) -> SolarSpectrum:
    """
    Looks in the directory data/solar_spectra/[year] for the file sorce_ssi_l3.csv, which should contain columns
    named 'time (Julian Date)', 'wavelength (nm)' and 'irradiance (W/m^2/nm)'
    """
    year = t.to_datetime().date().year  # type: ignore
    spectrum_path = pathlib.Path(
        "data/solar_spectra/sorce/" + str(year) + "/sorce_ssi_l3.csv"
    )
    ss = read_solar_spectrum_sorce(spectrum_path, t)

    # fix the units to be the same as those given from the file read by read_solar_spectrum
    ss.irradiances = ss.irradiances * 100

    return ss


def solar_count_rate_in_filter(
    solar_spectrum_path: pathlib.Path,
    solar_spectrum_time: Time,
    effective_area_path: pathlib.Path,
) -> float:
    """
    use effective area and theoretical spectra to calculate count rate in a filter due to solar activity
    through a convolution of the solar spectrum with the filter effective area
    """
    # large enough for beta to converge on its proper value
    num_interpolation_lambdas = 1000

    # read each file
    solar_spectrum = read_fixed_solar_spectrum(solar_spectrum_path)
    ea_data = read_effective_area(effective_area_path=effective_area_path)

    ea_lambdas = ea_data.lambdas
    ea_responses = ea_data.responses

    solar_lambdas = solar_spectrum.lambdas
    solar_irradiances = solar_spectrum.irradiances
    if len(solar_lambdas) == 0:
        print(f"Could not load solar spectrum data for {solar_spectrum_time}!")
        return 0.0

    # pick new set of lambdas to do the convolution over - the spectrum's range of wavelengths is much larger than the filter, so
    # the filter's wavelengths will determine the bounds of the integration
    lambdas, dlambda = np.linspace(
        np.min(ea_lambdas),
        np.max(ea_lambdas),
        endpoint=True,
        num=num_interpolation_lambdas,
        retstep=True,
    )

    # interpolate solar spectrum on new lambdas
    solar_irradiances_interpolation = interp1d(solar_lambdas, solar_irradiances)
    solar_irradiances_on_filter_lambdas = solar_irradiances_interpolation(lambdas)

    # interpolate responses on new lambdas
    ea_response_interpolation = interp1d(ea_lambdas, ea_responses)
    ea_responses_on_lambdas = ea_response_interpolation(lambdas)

    # assemble columns of [lambdas, irradiances, responses]
    spec = np.c_[
        np.c_[lambdas, solar_irradiances_on_filter_lambdas.T], ea_responses_on_lambdas.T
    ]

    # TODO: magic numbers
    cr = (
        np.sum(spec[:, 0] * spec[:, 1] * spec[:, 2]) * dlambda * 1e7 * 5.034116651114543
    )

    return cr
