from typing import TypeAlias

import numpy as np

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.error_propogation import ValueAndStandardDev


OHFlux: TypeAlias = ValueAndStandardDev


def OH_flux_from_count_rate(
    uw1: CountRate,
    uvv: CountRate,
    beta: DustReddeningPercent,
) -> OHFlux:
    # this comes from an OH spectral model in Bodewits et. al 2019 by convolving the OH spectrum through the uw1 filter
    # to convert count rate to flux
    alpha = 1.2750906353215913e-12

    oh_flux = alpha * (uw1.value - beta * uvv.value)

    oh_flux_err = alpha * np.sqrt(uw1.sigma**2 + (uvv.sigma * beta) ** 2)

    return OHFlux(value=oh_flux, sigma=oh_flux_err)
