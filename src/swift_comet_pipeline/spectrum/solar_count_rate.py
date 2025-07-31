from functools import cache

import numpy as np

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.spectrum.solar_spectrum import (
    get_solar_spectrum,
    interpolate_solar_spectrum_onto,
)
from swift_comet_pipeline.swift.read_filter_effective_area import (
    interpolate_filter_effective_area_onto,
    read_filter_effective_area,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.swift_filter import SwiftFilter


@cache
def solar_count_rate_in_filter_1au(filter_type: SwiftFilter) -> CountRate:

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    # large enough for beta to converge on its proper value - determined empirically
    num_interpolation_lambdas = 1000

    # solar_spectrum = read_fixed_solar_spectrum(spc.solar_spectrum_path)
    solar_spectrum = get_solar_spectrum()
    filter_effective_area = read_filter_effective_area(filter_type=filter_type)
    if filter_effective_area is None:
        return CountRate(value=np.nan, sigma=np.nan)

    # pick new set of lambdas to do the convolution over - the spectrum's range of wavelengths is much larger than the filter, so
    # the filter's wavelengths will determine the bounds of the integration
    lambdas_nm = np.linspace(
        np.min(filter_effective_area.lambdas_nm),
        np.max(filter_effective_area.lambdas_nm),
        endpoint=True,
        num=num_interpolation_lambdas,
    )

    solar_spectrum_interpolated = interpolate_solar_spectrum_onto(
        solar_spectrum=solar_spectrum, lambdas_nm=lambdas_nm
    )
    effective_area_interpolated = interpolate_filter_effective_area_onto(
        effective_area=filter_effective_area, lambdas_nm=lambdas_nm
    )
    count_rate_value = np.trapezoid(
        lambdas_nm
        * solar_spectrum_interpolated.spectral_irradiances_Wm2_nm
        * effective_area_interpolated.responses_cm2,
        lambdas_nm,
    )

    # Convert the solar flux into number of photons per second with factor (lambda/hc)
    # 1/hc ~ 5.034116651114543e24 * (1/Joule-meters)
    # the wavelengths are in nanometers, so we have another factor of 1e-9 to convert to meters
    # for a total factor of 5.034116651114543e15

    # multiply by 1/hc and convert cm2 to m2
    count_rate_value = count_rate_value * (5.034116651114543e15 / 10000)

    return CountRate(value=count_rate_value, sigma=0.0)
