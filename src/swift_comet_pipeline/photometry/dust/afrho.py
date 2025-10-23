import astropy.units as u

from swift_comet_pipeline.modeling.magnitude import magnitude_from_count_rate
from swift_comet_pipeline.modeling.spectra.solar.solar_count_rate import (
    solar_count_rate_in_filter_1au,
)
from swift_comet_pipeline.scp_types.compound.magnitude import Magnitude
from swift_comet_pipeline.scp_types.primitive import *


def calculate_afrho_in_cm(
    delta: u.Quantity,
    rh: u.Quantity,
    rho: u.Quantity,
    mag_in_filter: Magnitude,
    filter_type: UvotFilter,
) -> ValueAndStandardDev:

    # TODO: cite this

    solar_count_rate = solar_count_rate_in_filter_1au(filter_type=filter_type)
    solar_mag_1au = magnitude_from_count_rate(
        count_rate=solar_count_rate, filter_type=filter_type
    )

    mag_exponent = 0.4 * (solar_mag_1au.value - mag_in_filter.value)
    mag_exponent_err = 0.4 * mag_in_filter.sigma
    mag_exponent_var = mag_exponent_err**2

    afrho = float(4 * ((rh.to_value(u.AU) * delta.to_value(u.cm)) ** 2 * 10**mag_exponent) / rho.to_value(u.cm))  # type: ignore
    afrho_var = afrho * (0.4**2) * np.log(np.abs(mag_exponent)) * mag_exponent_var
    afrho_err = np.sqrt(afrho_var)

    return ValueAndStandardDev(value=afrho, sigma=afrho_err)


# def calculate_afrho(
#     delta: u.Quantity,
#     rh: u.Quantity,
#     rho: u.Quantity,
#     mag_in_filter: float,
#     filter_type: UvotFilter,
# ) -> u.Quantity:
#
#     # TODO: cite this
#
#     # get the magnitude of solar spectrum run through the uvv filter
#     # solar_uvv_count_rate_1au = solar_count_rate_in_filter_1au(UvotFilter.uvv)
#     # solar_uvv_mag_1au = magnitude_from_count_rate(
#     #     count_rate=solar_uvv_count_rate_1au, filter_type=UvotFilter.uvv
#     # )
#
#     solar_count_rate = solar_count_rate_in_filter_1au(filter_type=filter_type)
#     solar_mag_1au = magnitude_from_count_rate(
#         count_rate=solar_count_rate, filter_type=filter_type
#     )
#
#     # TODO: error propogation
#
#     mag_exponent = 0.4 * (solar_mag_1au.value - mag_in_filter)
#     afrho = 4 * ((rh.to_value(u.AU) * delta) ** 2 * 10**mag_exponent) / rho  # type: ignore
#
#     return afrho


# # TODO: remove old code
# def calculate_afrho_uvv(
#     delta: u.Quantity, rh: u.Quantity, rho: u.Quantity, magnitude_uvv: float
# ) -> u.Quantity:
#
#     # TODO: cite this
#
#     # get the magnitude of solar spectrum run through the uvv filter
#     solar_uvv_count_rate_1au = solar_count_rate_in_filter_1au(UvotFilter.uvv)
#     solar_uvv_mag_1au = magnitude_from_count_rate(
#         count_rate=solar_uvv_count_rate_1au, filter_type=UvotFilter.uvv
#     )
#
#     mag_exponent = 0.4 * (solar_uvv_mag_1au.value - magnitude_uvv)
#     afrho = 4 * ((rh.to_value(u.AU) * delta) ** 2 * 10**mag_exponent) / rho  # type: ignore
#
#     return afrho
