from functools import cache

import numpy as np

from swift_comet_pipeline.swift.swift_filter_parameters import get_filter_parameters
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.magnitude import Magnitude
from swift_comet_pipeline.types.swift_filter import SwiftFilter

_magnitude_error_factor = 2.5 / np.log(10)


@cache
def magnitude_from_count_rate(
    count_rate: CountRate, filter_type: SwiftFilter
) -> Magnitude:
    """
    Use the zero-point of the given swift filter to convert a count rate into magnitude
    """

    swift_filter_params = get_filter_parameters(filter_type=filter_type)
    assert swift_filter_params is not None

    mag = swift_filter_params.zero_point - 2.5 * np.log10(count_rate.value)
    mag_err = _magnitude_error_factor * count_rate.sigma / count_rate.value
    zp_err = swift_filter_params.zero_point_err
    sigma = np.sqrt(mag_err**2 + zp_err**2)

    return Magnitude(value=mag, sigma=sigma)


# TODO: clean up

# def magnitude_from_count_rate_1au(
#     count_rate: CountRate, filter_type: SwiftFilter
# ) -> Magnitude:
#
#     solar_counts = solar_count_rate_in_filter_1au(filter_type=filter_type)
#     mag_val = -2.5 * np.log10(count_rate.value / solar_counts.value)
#     mag_sig = _magnitude_error_factor * (count_rate.sigma / count_rate.value)
#     return Magnitude(value=mag_val, sigma=mag_sig)


# def magnitude_from_count_rate(
#     count_rate: CountRate, filter_type: SwiftFilter
# ) -> Magnitude:
#     """
#     Use the zero-point of the given swift filter to convert a count rate into magnitude
#     """
#
#     filter_params = get_filter_parameters(filter_type=filter_type)
#
#     mag = filter_params["zero_point"] - 2.5 * np.log10(count_rate.value)
#     mag_err = 2.5 * count_rate.sigma / (count_rate.value * np.log(10))
#     zp_err = filter_params["zero_point_err"]
#     sigma = np.sqrt(mag_err**2 + zp_err**2)
#
#     return Magnitude(value=mag, sigma=sigma)
