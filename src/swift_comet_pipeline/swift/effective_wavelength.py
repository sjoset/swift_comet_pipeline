from functools import cache

import numpy as np
import astropy.units as u

from swift_comet_pipeline.spectrum.solar_spectrum import (
    get_solar_spectrum,
    interpolate_solar_spectrum_onto,
)
from swift_comet_pipeline.swift.read_filter_effective_area import (
    interpolate_filter_effective_area_onto,
    read_filter_effective_area,
)
from swift_comet_pipeline.types.swift_filter import SwiftFilter


@cache
def average_wavelength_of_filter(filter_type: SwiftFilter) -> u.Quantity:

    filter_effective_area = read_filter_effective_area(filter_type=filter_type)
    if filter_effective_area is None:
        return 0 * u.nm  # type: ignore

    weighted_response = np.trapezoid(
        filter_effective_area.responses_cm2 * filter_effective_area.lambdas_nm,
        filter_effective_area.lambdas_nm,
    )
    total_response = np.trapezoid(
        filter_effective_area.responses_cm2, filter_effective_area.lambdas_nm
    )

    return (weighted_response / total_response) * u.nm  # type: ignore


@cache
def pivot_wavelength_of_filter(filter_type: SwiftFilter) -> u.Quantity:

    filter_effective_area = read_filter_effective_area(filter_type=filter_type)
    if filter_effective_area is None:
        return 0 * u.nm  # type: ignore

    piv_denom = np.trapezoid(
        filter_effective_area.responses_cm2 / filter_effective_area.lambdas_nm,
        filter_effective_area.lambdas_nm,
    )
    if piv_denom == 0.0:
        return 0 * u.nm  # type: ignore

    piv_num = np.trapezoid(
        filter_effective_area.lambdas_nm * filter_effective_area.responses_cm2,
        filter_effective_area.lambdas_nm,
    )

    pivot_wavelength = np.sqrt(piv_num / piv_denom) * u.nm  # type: ignore

    return pivot_wavelength


@cache
def effective_wavelength_of_filter_observing_solar_flux(
    filter_type: SwiftFilter,
) -> u.Quantity:
    # effective wavelength of filter while observing solar spectrum
    # photon-weighted for counting devices like UVOT

    filter_effective_area_raw = read_filter_effective_area(filter_type=filter_type)
    if filter_effective_area_raw is None:
        return 0 * u.nm  # type: ignore
    solar_spectrum_raw = get_solar_spectrum()

    num_interpolation_lambdas = 1000
    lambdas_nm = np.linspace(
        np.min(filter_effective_area_raw.lambdas_nm),
        np.max(filter_effective_area_raw.lambdas_nm),
        endpoint=True,
        num=num_interpolation_lambdas,
    )

    solar_spectrum = interpolate_solar_spectrum_onto(
        solar_spectrum=solar_spectrum_raw, lambdas_nm=lambdas_nm
    )
    filter_effective_area = interpolate_filter_effective_area_onto(
        effective_area=filter_effective_area_raw, lambdas_nm=lambdas_nm
    )

    eff_wave_numerator = np.trapezoid(
        filter_effective_area.responses_cm2
        * solar_spectrum.spectral_irradiances_Wm2_nm
        * lambdas_nm**2,
        lambdas_nm,
    )
    eff_wave_denominator = np.trapezoid(
        filter_effective_area.responses_cm2
        * solar_spectrum.spectral_irradiances_Wm2_nm
        * lambdas_nm,
        lambdas_nm,
    )

    return (eff_wave_numerator / eff_wave_denominator) * u.nm  # type: ignore
