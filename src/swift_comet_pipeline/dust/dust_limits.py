from functools import cache

import astropy.units as u

from swift_comet_pipeline.swift.effective_wavelength import (
    effective_wavelength_of_filter_observing_solar_flux,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.swift_filter import SwiftFilter

"""
Our filters are separated by 2770 angstroms, so our % per 1000 angstroms needs to be capped to avoid negative count rates for either filter
as a result of our linear approximation extending too far

TODO: show derivation or cite after publishing
"""


@cache
def get_dust_redness_lower_limit() -> DustReddeningPercent:
    """
    Returns the minimum dust redness that the uvw1 and uvv filter pair can measure with a linear approximation for the spectral slope
    """

    l_uvw1 = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uw1)
    l_uvv = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uvv)
    dlambda = l_uvv - l_uvw1

    min_redness = -20000.0 / dlambda.to_value(u.nm)  # type: ignore

    return DustReddeningPercent(min_redness)


@cache
def get_dust_redness_upper_limit() -> DustReddeningPercent:
    """
    Returns the maximum dust redness that the uvw1 and uvv filter pair can measure with a linear approximation for the spectral slope
    Redness is in percent per 100 nm or 1000 angstroms
    """

    l_uvw1 = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uw1)
    l_uvv = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uvv)
    dlambda = l_uvv - l_uvw1

    max_redness = 20000 / dlambda.to_value(u.nm)  # type: ignore

    return DustReddeningPercent(max_redness)
