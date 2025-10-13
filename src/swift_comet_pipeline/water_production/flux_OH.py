from functools import cache
from typing import TypeAlias

import astropy.units as u

from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive import *


# in erg/(s cm^2)
OHFlux: TypeAlias = ValueAndStandardDev


@cache
def OH_count_rates_to_flux_factor() -> u.Quantity:
    # this comes from an OH spectral model in Bodewits et. al 2019, via convolving the OH spectrum through the uw1 filter
    # to relate count rate to flux, in ergs/(cm**2  second)
    return 1.2750906353215913e-12 * u.erg / (u.cm**2 * u.s)  # type: ignore


# TODO: remove old code
# def OH_flux_from_count_rate(
#     uw1: CountRate,
#     uvv: CountRate,
#     beta: DustReddeningPercent,
# ) -> OHFlux:
#
#     alpha = OH_count_rates_to_flux_factor().to_value(u.erg / (u.cm**2 * u.s))  # type: ignore
#     oh_flux = alpha * (uw1 - beta * uvv)
#
#     return OHFlux(value=oh_flux.value, sigma=oh_flux.sigma)


def oh_flux_from_oh_count_rate(oh_cr: CountRate) -> OHFlux:

    alpha = OH_count_rates_to_flux_factor().to_value(u.erg / (u.cm**2 * u.s))  # type: ignore
    return alpha * oh_cr
