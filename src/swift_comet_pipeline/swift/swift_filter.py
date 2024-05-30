import pathlib
from enum import StrEnum, auto
from typing import Dict
from dataclasses import dataclass

import numpy as np
from astropy.io import fits


class SwiftFilter(StrEnum):
    uuu = auto()
    ubb = auto()
    uvv = auto()
    uw1 = auto()
    uw2 = auto()
    um2 = auto()
    white = auto()
    vgrism = auto()
    ugrism = auto()
    magnifier = auto()
    blocked = auto()
    unknown = auto()

    @classmethod
    def all_filters(cls):
        return [x for x in cls]


# how the filter influences image file names
filter_to_file_string_dict = {
    SwiftFilter.uuu: "uuu",
    SwiftFilter.ubb: "ubb",
    SwiftFilter.uvv: "uvv",
    SwiftFilter.uw1: "uw1",
    SwiftFilter.uw2: "uw2",
    SwiftFilter.um2: "um2",
    SwiftFilter.white: "uwh",
    SwiftFilter.vgrism: "ugv",
    SwiftFilter.ugrism: "ugu",
    SwiftFilter.magnifier: "umg",
    SwiftFilter.blocked: "ubl",
    SwiftFilter.unknown: "uun",
}


def filter_to_string(filter_type: SwiftFilter) -> str:
    return filter_to_file_string(filter_type)


def filter_to_file_string(filter_type: SwiftFilter) -> str:
    return filter_to_file_string_dict[filter_type]


def file_string_to_filter(filter_str: str) -> SwiftFilter:
    inverse_dict = {v: k for k, v in filter_to_file_string_dict.items()}
    return inverse_dict[filter_str]


# TODO: look these up and finish this
# TODO: verify each of these
# how the filter is represented as a string in the FITS file headers and the observation log
filter_to_obs_string_dict = {
    SwiftFilter.uuu: "U",
    SwiftFilter.ubb: "B",
    SwiftFilter.uvv: "V",
    SwiftFilter.uw1: "UVW1",
    SwiftFilter.uw2: "UVW2",
    SwiftFilter.um2: "UVM2",
    # TODO: check uwh
    SwiftFilter.white: "UWH",
    SwiftFilter.vgrism: "VGRISM",
    SwiftFilter.ugrism: "UGRISM",
    # TODO: check umg, ubl, uun
    SwiftFilter.magnifier: "UMG",
    SwiftFilter.blocked: "UBL",
    SwiftFilter.unknown: "UUN",
}


def filter_to_obs_string(filter_type: SwiftFilter) -> str:
    """description of how the FITS file headers denote which filter was used for taking the image"""

    return filter_to_obs_string_dict[filter_type]


def obs_string_to_filter(filter_str: str) -> SwiftFilter:
    inverse_dict = {v: k for k, v in filter_to_obs_string_dict.items()}
    return inverse_dict[filter_str]


# TODO: cite these from the swift documentation
# TODO: look up what 'cf' stands for
# TODO: Make SwiftFilterParameters a dataclass
# TODO: these are all technically a function of time, so we should incorporate that as an entry in the dataclass
def get_filter_parameters(filter_type: SwiftFilter) -> Dict:
    filter_params = {
        SwiftFilter.uvv: {
            "fwhm": 769,
            "zero_point": 17.89,
            "zero_point_err": 0.013,
            "cf": 2.61e-16,
            "cf_err": 2.4e-18,
        },
        SwiftFilter.ubb: {
            "fwhm": 975,
            "zero_point": 19.11,
            "zero_point_err": 0.016,
            "cf": 1.32e-16,
            "cf_err": 9.2e-18,
        },
        SwiftFilter.uuu: {
            "fwhm": 785,
            "zero_point": 18.34,
            "zero_point_err": 0.020,
            "cf": 1.5e-16,
            "cf_err": 1.4e-17,
        },
        SwiftFilter.uw1: {
            "fwhm": 693,
            "zero_point": 17.49,
            "zero_point_err": 0.03,
            "cf": 4.3e-16,
            "cf_err": 2.1e-17,
        },
        SwiftFilter.um2: {
            "fwhm": 498,
            "zero_point": 16.82,
            "zero_point_err": 0.03,
            "cf": 7.5e-16,
            "cf_err": 1.1e-17,
        },
        SwiftFilter.uw2: {
            "fwhm": 657,
            "zero_point": 17.35,
            "zero_point_err": 0.04,
            "cf": 6.0e-16,
            "cf_err": 6.4e-17,
        },
    }
    return filter_params[filter_type]


# TODO: units!
@dataclass
class FilterEffectiveArea:
    lambdas: np.ndarray
    responses: np.ndarray


def read_effective_area(effective_area_path: pathlib.Path) -> FilterEffectiveArea:
    # TODO: tag with astropy units, convert later? too slow?
    # or rename to ea_lambdas_nm
    filter_fits_hdul = fits.open(effective_area_path)
    filter_ea_data = filter_fits_hdul[1].data  # type: ignore
    ea_lambdas = (filter_ea_data["WAVE_MIN"] + filter_ea_data["WAVE_MAX"]) / 2
    # wavelengths are given in angstroms
    # ea_lambdas = ea_lambdas
    # wavelengths are given in angstroms: convert to nm
    ea_lambdas = ea_lambdas / 10
    ea_responses = filter_ea_data["SPECRESP"]

    # handle some lambda values repeating (their corresponding responses are 0, which is wrong, so throw them out)
    # Construct new list of unique lambdas, and the responses are now a list because we see some twice.
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

    filter_fits_hdul.close()

    return FilterEffectiveArea(
        lambdas=np.array(new_lambdas), responses=np.array(new_responses)
    )
