from swift_comet_pipeline.types.swift_filter import SwiftFilter


# how the filter influences image file names
_filter_to_file_string_dict = {
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


# # TODO: don't really need this one - clean it up if it's used anywhere
# # because we should just be explicit about what type of string we want from a SwiftFilter
# def filter_to_string(filter_type: SwiftFilter) -> str:
#     return filter_to_file_string(filter_type)


def filter_to_file_string(filter_type: SwiftFilter) -> str:
    return _filter_to_file_string_dict[filter_type]


def file_string_to_filter(filter_str: str) -> SwiftFilter:
    inverse_dict = {v: k for k, v in _filter_to_file_string_dict.items()}
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
