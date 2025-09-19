from swift_comet_pipeline.scp_types.primitive import *


# how the filter influences image file names
_filter_to_file_string_dict = {
    UvotFilter.uuu: "uuu",
    UvotFilter.ubb: "ubb",
    UvotFilter.uvv: "uvv",
    UvotFilter.uw1: "uw1",
    UvotFilter.uw2: "uw2",
    UvotFilter.um2: "um2",
    UvotFilter.white: "uwh",
    UvotFilter.vgrism: "ugv",
    UvotFilter.ugrism: "ugu",
    UvotFilter.magnifier: "umg",
    UvotFilter.blocked: "ubl",
    UvotFilter.unknown: "uun",
}


def filter_to_file_string(filter_type: UvotFilter) -> str:
    # how the filter is represented in image file names
    return _filter_to_file_string_dict[filter_type]


def file_string_to_filter(filter_str: str) -> UvotFilter:
    inverse_dict = {v: k for k, v in _filter_to_file_string_dict.items()}
    return inverse_dict[filter_str]


# TODO: look these up and finish this
# TODO: verify each of these
# how the filter is represented as a string in the FITS file headers and the observation log
filter_to_obs_string_dict = {
    UvotFilter.uuu: "U",
    UvotFilter.ubb: "B",
    UvotFilter.uvv: "V",
    UvotFilter.uw1: "UVW1",
    UvotFilter.uw2: "UVW2",
    UvotFilter.um2: "UVM2",
    # TODO: check uwh
    UvotFilter.white: "UWH",
    UvotFilter.vgrism: "VGRISM",
    UvotFilter.ugrism: "UGRISM",
    # TODO: check umg, ubl, uun
    UvotFilter.magnifier: "UMG",
    UvotFilter.blocked: "UBL",
    UvotFilter.unknown: "UUN",
}


def filter_to_obs_string(filter_type: UvotFilter) -> str:
    """description of how the FITS file headers denote which filter was used for taking the image"""

    return filter_to_obs_string_dict[filter_type]


def obs_string_to_filter(filter_str: str) -> UvotFilter:
    inverse_dict = {v: k for k, v in filter_to_obs_string_dict.items()}
    return inverse_dict[filter_str]
