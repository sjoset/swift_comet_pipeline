import pathlib

from astropy.time import Time

from solar_spectrum import solar_count_rate_in_filter
from reddening_correction import reddening_correction, DustReddeningPercent
from aperture_photometry import AperturePhotometryResult

from typing import Tuple

__version__ = "0.0.1"


__all__ = ["OH_flux_from_count_rate", "OH_flux_from_count_rate_fixed_beta"]


def OH_flux_from_count_rate(
    solar_spectrum_path: pathlib.Path,
    solar_spectrum_time: Time,
    effective_area_uw1_path: pathlib.Path,
    effective_area_uvv_path: pathlib.Path,
    result_uw1: AperturePhotometryResult,
    result_uvv: AperturePhotometryResult,
    dust_redness: DustReddeningPercent,
) -> Tuple[float, float]:
    # TODO: read Lucy's thesis and figure out why the conversion units are 1.275e-12
    alpha = 1.2750906353215913e-12

    solar_count_rate_in_uw1 = solar_count_rate_in_filter(
        solar_spectrum_path=solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=effective_area_uw1_path,
    )
    solar_count_rate_in_uvv = solar_count_rate_in_filter(
        solar_spectrum_path=solar_spectrum_path,
        solar_spectrum_time=solar_spectrum_time,
        effective_area_path=effective_area_uvv_path,
    )

    # print(f"solar count rate in uw1: {solar_count_rate_in_uw1}")
    # print(f"solar count rate in uvv: {solar_count_rate_in_uvv}")
    beta_pre_reddening = solar_count_rate_in_uw1 / solar_count_rate_in_uvv
    beta = (
        reddening_correction(
            effective_area_uw1_path=effective_area_uw1_path,
            effective_area_uvv_path=effective_area_uvv_path,
            dust_redness=dust_redness,
        )
        * beta_pre_reddening
    )

    oh_flux = alpha * (result_uw1.net_count_rate - beta * result_uvv.net_count_rate)

    # TODO: cr_to_flux.py -> error_prop
    # propogate error for beta * count_rate_uvv
    # propogate that into oh_count_rate

    return oh_flux, beta


def OH_flux_from_count_rate_fixed_beta(
    # solar_spectrum_path: pathlib.Path,
    # solar_spectrum_time: Time,
    effective_area_uw1_path: pathlib.Path,
    effective_area_uvv_path: pathlib.Path,
    result_uw1: AperturePhotometryResult,
    result_uvv: AperturePhotometryResult,
    dust_redness: DustReddeningPercent,
) -> Tuple[float, float]:
    """get OH flux from OH cr"""
    # beta = 0.09276191501510327
    beta = 0.1043724648186691
    beta = (
        reddening_correction(
            effective_area_uw1_path=effective_area_uw1_path,
            effective_area_uvv_path=effective_area_uvv_path,
            dust_redness=dust_redness,
        )
        * beta
    )

    # cr_ref_uw1 = beta * result_uvv.net_count_rate
    # cr_ref_uw1_err = beta * cr_v_err
    cr_OH = result_uw1.net_count_rate - beta * result_uvv.net_count_rate
    # cr_OH_err = error_prop("sub", cr_uw1, cr_uw1_err, cr_ref_uw1, cr_ref_uw1_err)

    flux_OH = cr_OH * 1.2750906353215913e-12
    # flux_OH_err = cr_OH_err * 1.2750906353215913e-12

    # flux_uw1, flux_uw1_err = flux_ref_uw1(
    #     spec_name_sun, spec_name_OH, cr_uw1, cr_uw1_err, cr_v, cr_v_err, r
    # )
    # flux_v, flux_v_err = flux_ref_v(
    #     spec_name_sun, spec_name_OH, cr_uw1, cr_uw1_err, cr_v, cr_v_err, r
    # )
    # if if_show == True:
    #     print(
    #         "flux of uw1 (reflection): " + str(flux_uw1) + " +/- " + str(flux_uw1_err)
    #     )
    #     print("flux of v: " + str(flux_v) + " +/- " + str(flux_v_err))
    # return flux_OH, flux_OH_err
    return flux_OH, beta
