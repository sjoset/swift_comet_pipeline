#!/usr/bin/env python3

import pathlib
import numpy as np

from astropy.io import fits

from dataclasses import dataclass


@dataclass
class FilterEffectiveArea:
    lambdas: np.ndarray
    responses: np.ndarray


__all__ = ["read_effective_area", "FilterEffectiveArea"]


def read_effective_area(effective_area_path: pathlib.Path) -> FilterEffectiveArea:
    # TODO: tag with astropy units, convert later?
    filter_ea_data = fits.open(effective_area_path)[1].data  # type: ignore
    ea_lambdas = (filter_ea_data["WAVE_MIN"] + filter_ea_data["WAVE_MAX"]) / 2
    # wavelengths are given in angstroms: convert to nm
    ea_lambdas = ea_lambdas / 10
    ea_responses = filter_ea_data["SPECRESP"]

    # handle some lambda values repeating (their corresponding responses are 0, which is wrong, so throw them out)
    # Contstruct new list of unique lambdas, and the responses are now a list because we see some twice.
    # The bad response values are 0.0, so we take the max between the 'good' value and this throwaway response value
    new_lambdas = []
    new_responses = []
    for lmbda, r in zip(ea_lambdas, ea_responses):
        if lmbda not in new_lambdas:
            new_lambdas.append(lmbda)
            new_responses.append([r])
        else:
            idx = new_lambdas.index(lmbda)
            new_responses[idx].append(r)
    new_responses = list(map(max, new_responses))

    return FilterEffectiveArea(
        lambdas=np.array(new_lambdas), responses=np.array(new_responses)
    )
