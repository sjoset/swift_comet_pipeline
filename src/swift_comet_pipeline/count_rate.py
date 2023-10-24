import numpy as np

from typing import TypeAlias

from error_propogation import ValueAndStandardDev
from swift_filter import SwiftFilter, get_filter_parameters

__all__ = ["CountRate", "CountRatePerPixel", "magnitude_from_count_rate"]


CountRate: TypeAlias = ValueAndStandardDev
CountRatePerPixel: TypeAlias = ValueAndStandardDev
Magnitude: TypeAlias = ValueAndStandardDev


# TODO: does this function belong here?
def magnitude_from_count_rate(
    count_rate: CountRate, filter_type: SwiftFilter
) -> Magnitude:
    filter_params = get_filter_parameters(filter_type=filter_type)

    mag = filter_params["zero_point"] - 2.5 * np.log10(count_rate.value)
    mag_err = 2.5 * count_rate.sigma / (count_rate.value * np.log(10))
    zp_err = filter_params["zero_point_err"]
    sigma = np.sqrt(mag_err**2 + zp_err**2)

    return Magnitude(value=mag, sigma=sigma)
