import numpy as np

from swift_comet_pipeline.swift.count_rate import CountRate
from swift_comet_pipeline.swift.magnitude import Magnitude
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, get_filter_parameters


def magnitude_from_count_rate(
    count_rate: CountRate, filter_type: SwiftFilter
) -> Magnitude:
    # TODO: document
    # TODO: is this valid for a single pixel, or overall count rate in an aperture?

    filter_params = get_filter_parameters(filter_type=filter_type)

    mag = filter_params["zero_point"] - 2.5 * np.log10(count_rate.value)
    mag_err = 2.5 * count_rate.sigma / (count_rate.value * np.log(10))
    zp_err = filter_params["zero_point_err"]
    sigma = np.sqrt(mag_err**2 + zp_err**2)

    return Magnitude(value=mag, sigma=sigma)
