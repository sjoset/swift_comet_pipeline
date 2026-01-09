from typing import TypeAlias

import astropy.units as u


from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive import *

# TODO: move to types
# in erg/(s cm^2)
OHFlux: TypeAlias = ValueAndStandardDev


# TODO: turn count_rates_to_flux into a [UvotFilter, float] dictionary based on response to OH fluorescence spectrum
# The response ratio of the UVW1/UVW2 filters indicate that we get ~0.2 counts per count in UVW1 - so 1 count is worth ~5 times the flux

# TODO: these units are wrong but they are effectively stripped anyway
_flux_units = u.erg / (u.cm**2 * u.s)  # type: ignore
_count_rates_to_flux_factor = {
    # this comes from an OH spectral model in Bodewits et. al 2019, via convolving the OH spectrum through the uw1 filter
    UvotFilter.uw1: 1.2750906353215913e-12 * _flux_units,
    # this was estimated based on the relative effective areas in a window around 308 nm
    # TODO: obtain OH spectrum and do convolution vs filter response for uw2 for something better than '5': it is ~5 times less sensitive at 308 nm
    UvotFilter.uw2: 5 * 1.2750906353215913e-12 * _flux_units,
    # TODO: obtain OH spectrum and do convolution vs filter response for uuu for something better than '1/6': it is ~6 times more sensitive at 308 nm
    UvotFilter.uuu: (1 / 6) * 1.2750906353215913e-12 * _flux_units,
    # TODO: obtain OH spectrum and do convolution vs filter response for um2 for something better than 0.013: it is 1.3% as sensitive at 308 nm compared to uw1
    UvotFilter.um2: (0.013) * 1.2750906353215913e-12 * _flux_units,
}


def oh_count_rates_to_flux_factor(filter_type: UvotFilter) -> u.Quantity:
    # TODO: log invalid filter lookups
    return _count_rates_to_flux_factor.get(filter_type, 0 * _flux_units)


def oh_flux_from_oh_count_rate(
    oh_count_rate: CountRate, filter_type: UvotFilter
) -> OHFlux:
    return (
        oh_count_rates_to_flux_factor(filter_type=filter_type).to_value(_flux_units)
        * oh_count_rate
    )
