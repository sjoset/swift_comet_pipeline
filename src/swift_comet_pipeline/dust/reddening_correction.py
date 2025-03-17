from functools import cache

import numpy as np

from swift_comet_pipeline.swift.read_filter_effective_area import (
    read_filter_effective_area,
)
from swift_comet_pipeline.types import DustReddeningPercent, SwiftFilter


__all__ = ["reddening_correction"]


@cache
def _pre_middle_factor() -> float:

    ea_data_uw1 = read_filter_effective_area(filter_type=SwiftFilter.uw1)
    ea_data_uvv = read_filter_effective_area(filter_type=SwiftFilter.uvv)
    if ea_data_uw1 is None or ea_data_uvv is None:
        return np.nan

    uw1_lambdas = ea_data_uw1.lambdas_nm
    uw1_responses = ea_data_uw1.responses_cm2

    uvv_lambdas = ea_data_uvv.lambdas_nm
    uvv_responses = ea_data_uvv.responses_cm2

    wave_uw1 = 0
    ea_uw1 = 0
    wave_v = 0
    ea_v = 0

    # TODO: rewrite this without loops
    delta_wave_uw1 = uw1_lambdas[1] - uw1_lambdas[0]
    delta_wave_v = uvv_lambdas[1] - uvv_lambdas[0]
    for i in range(len(uw1_lambdas)):
        wave_uw1 += uw1_lambdas[i] * uw1_responses[i] * delta_wave_uw1
        ea_uw1 += uw1_responses[i] * delta_wave_uw1
    wave_uw1 = wave_uw1 / ea_uw1
    for i in range(len(uvv_lambdas)):
        wave_v += uvv_lambdas[i] * uvv_responses[i] * delta_wave_v
        ea_v += uvv_responses[i] * delta_wave_v
    wave_v = wave_v / ea_v

    # TODO: magic number
    # TODO: get reddening correction factor: do this with proper units (when EA lambdas are in angstroms, this is 200000)
    pre_middle_factor = (wave_v - wave_uw1) / 20000

    # multiply by dust_redness in percent per 100 nm to get the 'middle factor' in reddening_correction()
    return pre_middle_factor


@cache
def reddening_correction(dust_redness: DustReddeningPercent) -> float:
    """
    get the correction factor of beta for dust reddening
    units of reddening: %/100nm

    where beta is the factor in (uw1 - beta * uvv)
    """

    middle_factor = _pre_middle_factor() * dust_redness

    # Reference: eq. 3.36 and 3.39 in Xing thesis
    return (1 - middle_factor) / (1 + middle_factor)
