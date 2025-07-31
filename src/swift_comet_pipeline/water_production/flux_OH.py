from typing import TypeAlias

import astropy.units as u

from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.error_propogation import ValueAndStandardDev


# in erg/(s cm^2)
OHFlux: TypeAlias = ValueAndStandardDev


def OH_count_rates_to_flux_factor() -> u.Quantity:
    # this comes from an OH spectral model in Bodewits et. al 2019, via convolving the OH spectrum through the uw1 filter
    # to relate count rate to flux, in ergs/(cm**2  second)
    return 1.2750906353215913e-12 * u.erg / (u.cm**2 * u.s)  # type: ignore


def OH_flux_from_count_rate(
    uw1: CountRate,
    uvv: CountRate,
    beta: DustReddeningPercent,
) -> OHFlux:

    alpha = OH_count_rates_to_flux_factor().to_value(u.erg / (u.cm**2 * u.s))  # type: ignore
    oh_flux = alpha * (uw1 - beta * uvv)

    return OHFlux(value=oh_flux.value, sigma=oh_flux.sigma)
