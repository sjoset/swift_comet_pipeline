import astropy.units as u

from swift_comet_pipeline.spectrum.solar_count_rate import (
    solar_count_rate_in_filter_1au,
)
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.types.swift_filter import SwiftFilter


def calculate_afrho(
    delta: u.Quantity, rh: u.Quantity, rho: u.Quantity, magnitude_uvv: float
) -> u.Quantity:

    # get the magnitude of solar spectrum run through the uvv filter
    solar_uvv_count_rate_1au = solar_count_rate_in_filter_1au(SwiftFilter.uvv)
    solar_uvv_mag_1au = magnitude_from_count_rate(
        count_rate=solar_uvv_count_rate_1au, filter_type=SwiftFilter.uvv
    )

    mag_exponent = 0.4 * (solar_uvv_mag_1au.value - magnitude_uvv)
    afrho = 4 * ((rh.to_value(u.AU) * delta) ** 2 * 10**mag_exponent) / rho  # type: ignore

    return afrho
