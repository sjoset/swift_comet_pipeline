from functools import cache

import numpy as np
from scipy.interpolate import interp1d

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.spectrum.solar_spectrum import read_fixed_solar_spectrum
from swift_comet_pipeline.swift import read_filter_effective_area
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

    solar_spectrum = read_fixed_solar_spectrum(spc.solar_spectrum_path)
    ea_data = read_filter_effective_area(filter_type=filter_type)
    if ea_data is None:
        return CountRate(value=np.nan, sigma=np.nan)

    # effective area: convert units
    ea_lambdas = ea_data.lambdas_nm
    ea_responses_m2 = ea_data.responses_cm2 / 10000

    solar_lambdas = solar_spectrum.lambdas_nm
    solar_irradiances = solar_spectrum.spectral_irradiances_Wm2_nm
    if len(solar_lambdas) == 0:
        print(f"Could not load solar spectrum data!")
        return CountRate(value=np.nan, sigma=np.nan)

    # pick new set of lambdas to do the convolution over - the spectrum's range of wavelengths is much larger than the filter, so
    # the filter's wavelengths will determine the bounds of the integration
    lambdas, dlambda = np.linspace(
        np.min(ea_lambdas),
        np.max(ea_lambdas),
        endpoint=True,
        num=num_interpolation_lambdas,
        retstep=True,
    )

    # interpolate solar spectrum on new lambdas
    solar_irradiances_interpolation = interp1d(solar_lambdas, solar_irradiances)
    solar_irradiances_on_filter_lambdas = solar_irradiances_interpolation(lambdas)

    # interpolate responses on new lambdas
    ea_response_interpolation = interp1d(ea_lambdas, ea_responses_m2)
    ea_responses_on_lambdas = ea_response_interpolation(lambdas)

    # Convert the solar flux into number of photons per second with factor (lambda/hc)
    # 1/hc ~ 5.034116651114543e24 * (1/Joule-meters)
    # the wavelengths are in nanometers, so we have another factor of 1e-9 to convert to meters
    # for a total factor of 5.034116651114543e15

    # total number of photons seen in the filter - integral of I(lambda) * A(lambda) * (lambda/hc) dlambda
    cr = (
        np.sum(lambdas * solar_irradiances_on_filter_lambdas * ea_responses_on_lambdas)
        * dlambda
        * 5.034116651114543e15
    )

    return CountRate(value=cr, sigma=0.0)
