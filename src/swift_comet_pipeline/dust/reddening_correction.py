import pathlib
from typing import TypeAlias

from swift_comet_pipeline.swift.swift_filter import read_effective_area

DustReddeningPercent: TypeAlias = float


def reddening_correction(
    effective_area_uw1_path: pathlib.Path,
    effective_area_uvv_path: pathlib.Path,
    dust_redness: DustReddeningPercent,
) -> float:
    """
    get the correction factor of beta for dust reddening
    units of reddening: %/100nm

    where beta is the factor in (uw1 - beta * uvv)
    """

    ea_data_uw1 = read_effective_area(effective_area_path=effective_area_uw1_path)
    uw1_lambdas = ea_data_uw1.lambdas
    uw1_responses = ea_data_uw1.responses

    ea_data_uvv = read_effective_area(effective_area_path=effective_area_uvv_path)
    uvv_lambdas = ea_data_uvv.lambdas
    uvv_responses = ea_data_uvv.responses

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
    # get reddening correction factor: do this with proper units (when EA lambdas are in angstroms, this is 200000)
    middle_factor = (wave_v - wave_uw1) * dust_redness / 20000

    # Reference: eq. 3.36 and 3.39 in Xing thesis
    return (1 - middle_factor) / (1 + middle_factor)
