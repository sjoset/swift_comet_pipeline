from typing import TypeAlias

import numpy as np
from astropy.time import Time

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.spectrum.solar_spectrum import solar_count_rate_in_filter
from swift_comet_pipeline.dust.reddening_correction import (
    reddening_correction,
    DustReddeningPercent,
)
from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
from swift_comet_pipeline.swift.count_rate import CountRate

OHFlux: TypeAlias = ValueAndStandardDev


def beta_parameter(
    dust_redness: DustReddeningPercent, solar_spectrum_time: Time | None = None
) -> float:
    if solar_spectrum_time is None:
        # arbitrary date, happy birthday Mom
        solar_spectrum_time = Time("2016-08-08")

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    solar_count_rate_in_uw1 = solar_count_rate_in_filter(
        solar_spectrum_path=spc.solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=spc.effective_area_uw1_path,
    )
    solar_count_rate_in_uvv = solar_count_rate_in_filter(
        solar_spectrum_path=spc.solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=spc.effective_area_uvv_path,
    )

    beta_pre_reddening = solar_count_rate_in_uw1 / solar_count_rate_in_uvv
    beta = (
        reddening_correction(
            effective_area_uw1_path=spc.effective_area_uw1_path,
            effective_area_uvv_path=spc.effective_area_uvv_path,
            dust_redness=dust_redness,
        )
        * beta_pre_reddening
    )

    return beta


def OH_flux_from_count_rate(
    uw1: CountRate,
    uvv: CountRate,
    beta: float,
) -> OHFlux:
    # this comes from an OH spectral model in Bodewits et. al 2019
    alpha = 1.2750906353215913e-12

    oh_flux = alpha * (uw1.value - beta * uvv.value)

    # TODO: check which of these is correct - ask Lucy
    oh_flux_err = alpha * np.sqrt(uw1.sigma**2 + (uvv.sigma * beta) ** 2)
    # oh_flux_err = alpha * np.sqrt(uw1.sigma**2 + np.abs(beta) * uvv.sigma**2)

    return OHFlux(value=oh_flux, sigma=oh_flux_err)
