import numpy as np

from swift_comet_pipeline.swift.swift_filter_parameters import get_filter_parameters
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.magnitude import Magnitude
from swift_comet_pipeline.types.swift_filter import SwiftFilter


def magnitude_from_count_rate(
    count_rate: CountRate, filter_type: SwiftFilter
) -> Magnitude:
    """
    Use the zero-point of the given swift filter to convert a count rate into magnitude
    """

    swift_filter_params = get_filter_parameters(filter_type=filter_type)
    assert swift_filter_params is not None

    mag = swift_filter_params.zero_point - 2.5 * np.log10(count_rate.value)
    mag_err = 2.5 * count_rate.sigma / (count_rate.value * np.log(10))
    zp_err = swift_filter_params.zero_point_err
    sigma = np.sqrt(mag_err**2 + zp_err**2)

    return Magnitude(value=mag, sigma=sigma)


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
