from functools import cache

import astropy.units as u


from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.swift.filters.filter_wavelengths import (
    effective_wavelength_of_filter_observing_solar_flux,
)


# TODO: re-enable cache
# @cache
def reddening_correction(
    dust_redness: DustReddeningPercent, filter_low: UvotFilter, filter_high: UvotFilter
) -> float:
    """
    get the correction factor of beta for dust reddening
    units of reddening: %/100nm

    where beta is the factor in (oh_filter - beta * dust_filter)

    TODO: document derivation or cite US10 paper
    """

    # l_uvw1 = effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uw1)
    # l_uvv = effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uvv)

    lambda_low = effective_wavelength_of_filter_observing_solar_flux(
        filter_type=filter_low
    )
    lambda_high = effective_wavelength_of_filter_observing_solar_flux(
        filter_type=filter_high
    )

    # dlambda_nm = (l_uvv - l_uvw1).to_value(u.nm)  # type: ignore
    dlambda_nm = (lambda_high - lambda_low).to_value(u.nm)  # type: ignore

    t = dust_redness * dlambda_nm / 20000

    return (1 - t) / (1 + t)
