from functools import cache

import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.types.filter_effective_area import FilterEffectiveArea
from swift_comet_pipeline.types.swift_filter import SwiftFilter


@cache
def read_filter_effective_area(filter_type: SwiftFilter) -> FilterEffectiveArea | None:

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    # this should be a function for FilterType -> pathlib.Path
    if filter_type == SwiftFilter.uw1:
        effective_area_path = spc.effective_area_uw1_path
    elif filter_type == SwiftFilter.uvv:
        effective_area_path = spc.effective_area_uvv_path
    else:
        return None

    filter_fits_hdul = fits.open(effective_area_path)
    filter_ea_data = filter_fits_hdul[1].data  # type: ignore
    ea_lambdas_angstroms = (filter_ea_data["WAVE_MIN"] + filter_ea_data["WAVE_MAX"]) / 2
    # wavelengths are given in angstroms: convert to nm
    ea_lambdas_nm = ea_lambdas_angstroms / 10
    ea_responses = filter_ea_data["SPECRESP"]

    # handle some lambda values repeating (their corresponding responses are 0, which is wrong, so throw them out)
    # Construct new list of unique lambdas, and the responses are now a list because we see some twice.
    # The bad response values are 0.0, so we take the max between the 'good' value and this throwaway response value
    # TODO: we can do this more efficiently
    new_lambdas = []
    new_responses = []
    for lmbda, r in zip(ea_lambdas_nm, ea_responses):
        if lmbda not in new_lambdas:
            new_lambdas.append(lmbda)
            new_responses.append([r])
        else:
            idx = new_lambdas.index(lmbda)
            new_responses[idx].append(r)
    new_responses = list(map(max, new_responses))

    filter_fits_hdul.close()

    return FilterEffectiveArea(
        lambdas_nm=np.array(new_lambdas), responses_cm2=np.array(new_responses)
    )


def interpolate_filter_effective_area_onto(
    effective_area: FilterEffectiveArea, lambdas_nm: np.ndarray
) -> FilterEffectiveArea:

    effective_areas_interpolation = interp1d(
        effective_area.lambdas_nm, effective_area.responses_cm2
    )
    responses_on_lambdas = effective_areas_interpolation(lambdas_nm)

    return FilterEffectiveArea(
        lambdas_nm=lambdas_nm.copy(), responses_cm2=responses_on_lambdas
    )
