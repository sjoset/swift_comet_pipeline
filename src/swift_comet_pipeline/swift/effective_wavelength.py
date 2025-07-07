from functools import cache

import numpy as np
import astropy.units as u

from swift_comet_pipeline.swift.read_filter_effective_area import (
    read_filter_effective_area,
)
from swift_comet_pipeline.types.swift_filter import SwiftFilter


@cache
def effective_wavelength_of_filter(filter_type: SwiftFilter):

    filter_effective_area = read_filter_effective_area(filter_type=filter_type)
    if filter_effective_area is None:
        return 0 * u.nm  # type: ignore

    total_response = np.trapezoid(
        filter_effective_area.responses_cm2, filter_effective_area.lambdas_nm
    )
    weighted_response = np.trapezoid(
        filter_effective_area.responses_cm2 * filter_effective_area.lambdas_nm,
        filter_effective_area.lambdas_nm,
    )

    # pivot wavelengths and effective wavelengths are very close in value, but if want to switch to using pivot:
    # piv_denom = np.trapezoid(
    #     filter_effective_area.responses_cm2 / filter_effective_area.lambdas_nm,
    #     filter_effective_area.lambdas_nm,
    # )
    # pivot_wavelength = np.sqrt(weighted_response/piv_denom) * u.nm

    return (weighted_response / total_response) * u.nm  # type: ignore
