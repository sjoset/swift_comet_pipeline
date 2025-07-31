from functools import cache

import astropy.units as u

from swift_comet_pipeline.swift.effective_wavelength import (
    effective_wavelength_of_filter_observing_solar_flux,
)
from swift_comet_pipeline.types import DustReddeningPercent, SwiftFilter


__all__ = ["reddening_correction"]


@cache
def reddening_correction(dust_redness: DustReddeningPercent) -> float:
    """
    get the correction factor of beta for dust reddening
    units of reddening: %/100nm

    where beta is the factor in (uw1 - beta * uvv)

    TODO: document derivation or cite US10 paper
    """

    l_uvw1 = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uw1)
    l_uvv = effective_wavelength_of_filter_observing_solar_flux(SwiftFilter.uvv)

    dlambda_nm = (l_uvv - l_uvw1).to_value(u.nm)  # type: ignore
    t = dust_redness * dlambda_nm / 20000

    return (1 - t) / (1 + t)
