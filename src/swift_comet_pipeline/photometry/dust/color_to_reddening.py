from enum import Enum, auto

import numpy as np

from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)


class JohnsonCousinsFilter(Enum):
    U = auto()
    B = auto()
    V = auto()
    R = auto()
    I = auto()


class SDSSFilter(Enum):
    u = auto()
    g = auto()
    r = auto()
    i = auto()
    z = auto()


# https://www.aavso.org/filters
_jc_wavelengths_angstrom = {
    JohnsonCousinsFilter.U: 3663,
    JohnsonCousinsFilter.B: 4361,
    JohnsonCousinsFilter.V: 5448,
    JohnsonCousinsFilter.R: 6407,
    JohnsonCousinsFilter.I: 7980,
}
_sdss_wavelengths_angstrom = {
    SDSSFilter.u: 3543,
    SDSSFilter.g: 4770,
    SDSSFilter.r: 6231,
    SDSSFilter.i: 7625,
    SDSSFilter.z: 9097,
}


def _get_filter_wavelength_angstrom(
    f: JohnsonCousinsFilter | SDSSFilter,
) -> float | None:
    if isinstance(f, JohnsonCousinsFilter):
        return _jc_wavelengths_angstrom.get(f, None)
    elif isinstance(f, SDSSFilter):
        return _sdss_wavelengths_angstrom.get(f, None)


class ColorMagnitudeFilterPair(Enum):
    # Johnson-Cousins
    U_B = (JohnsonCousinsFilter.U, JohnsonCousinsFilter.B)
    B_V = (JohnsonCousinsFilter.B, JohnsonCousinsFilter.V)
    V_R = (JohnsonCousinsFilter.V, JohnsonCousinsFilter.R)
    R_I = (JohnsonCousinsFilter.R, JohnsonCousinsFilter.I)
    V_I = (JohnsonCousinsFilter.V, JohnsonCousinsFilter.I)
    B_R = (JohnsonCousinsFilter.B, JohnsonCousinsFilter.R)
    B_I = (JohnsonCousinsFilter.B, JohnsonCousinsFilter.I)

    # SDSS ugriz (AB system)
    u_g = (SDSSFilter.u, SDSSFilter.g)
    g_r = (SDSSFilter.g, SDSSFilter.r)
    r_i = (SDSSFilter.r, SDSSFilter.i)
    i_z = (SDSSFilter.i, SDSSFilter.z)


# Ramirez et. al. 2012
_solar_colors_by_filter_pair = {
    ColorMagnitudeFilterPair.U_B: 0.158,
    ColorMagnitudeFilterPair.B_V: 0.653,
    ColorMagnitudeFilterPair.V_R: 0.356,
    ColorMagnitudeFilterPair.V_I: 0.701,
    # R_I = V_R - V_I
    ColorMagnitudeFilterPair.R_I: 0.356 - 0.701,
    ColorMagnitudeFilterPair.B_R: 1.005,
    # B_I = B_V + V_I
    ColorMagnitudeFilterPair.B_I: 0.653 + 0.701,
    ColorMagnitudeFilterPair.u_g: 1.43,
    ColorMagnitudeFilterPair.g_r: 0.44,
    ColorMagnitudeFilterPair.r_i: 0.11,
    ColorMagnitudeFilterPair.i_z: 0.03,
}


def color_to_reddening(
    color_mag: float, fp: ColorMagnitudeFilterPair, verbose: bool = False
) -> tuple[DustReddeningPercent, float] | None:

    solar_color = _solar_colors_by_filter_pair.get(fp)
    if solar_color is None:
        print(f"Could not find solar color for pair {fp}")
        return None

    filter_low = fp.value[0]
    wave_low_ang = _get_filter_wavelength_angstrom(f=filter_low)
    filter_high = fp.value[1]
    wave_high_ang = _get_filter_wavelength_angstrom(f=filter_high)
    if wave_low_ang is None or wave_high_ang is None:
        print(f"Could not find wavelengths of filter pair {fp}")
        return None

    # convert to nm
    wave_low = wave_low_ang / 10
    wave_high = wave_high_ang / 10

    delta_mag = color_mag - solar_color
    delta_lambda = np.abs(wave_high - wave_low)

    x = np.pow(10.0, delta_mag * 0.4)

    # normalized spectral gradient in 100% per 100 nm
    gamma = 20000

    reddening = ((x - 1) / (x + 1)) * gamma / delta_lambda
    mid_wave_nm = (wave_high + wave_low) / 2.0

    if verbose:
        print(f"Filters: {filter_low.name=}\t{filter_high.name=}")
        print(f"Lambdas: {wave_low=}\t{wave_high=}")
        print(f"Color mag: {color_mag}\tSolar color: {solar_color}")
        print(f"Delta lambda: {delta_lambda}")
        print(f"Delta mag: {delta_mag}")
        print(f"Reddening: {reddening}")
        print(f"Mid wavelength, nm: {mid_wave_nm}")

    return reddening, mid_wave_nm
