import pathlib
import numpy as np
import pandas as pd

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, TypeAlias, List, Tuple


__all__ = [
    "SwiftObservationLog",
    "SwiftUVOTImage",
    "SwiftStackedUVOTImage",
    "SwiftFilter",
    "SwiftUVOTImageType",
    "SwiftOrbitID",
    "SwiftObservationID",
    "swift_orbit_id_from_obsid",
    "swift_observation_id_from_int",
    "PixelCoord",
    "filter_to_file_string",
    "file_string_to_filter",
    "filter_to_obs_string",
    "obs_string_to_filter",
]


SwiftObservationLog: TypeAlias = pd.DataFrame
SwiftUVOTImage: TypeAlias = np.ndarray
SwiftObservationID: TypeAlias = str
SwiftOrbitID: TypeAlias = str


class SwiftPixelResolution(Enum):
    event_mode = 0.502
    data_mode = 1.0


class SwiftFilter(str, Enum):
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


def filter_to_file_string(filter_type: SwiftFilter) -> str:
    return filter_to_file_string_dict[filter_type]


def file_string_to_filter(filter_str: str) -> SwiftFilter:
    inverse_dict = {v: k for k, v in filter_to_file_string_dict.items()}
    return inverse_dict[filter_str]


# TODO: look these up and finish this
# TODO: verify each of these
# how the filter is represented in the FITS file headers and the observation log
filter_to_obs_string_dict = {
    SwiftFilter.uuu: "U",
    # SwiftFilter.ubb: "ubb",
    SwiftFilter.uvv: "V",
    SwiftFilter.uw1: "UVW1",
    SwiftFilter.uw2: "UVW2",
    SwiftFilter.um2: "UVM2",
    # SwiftFilter.white: "uwh",
    # SwiftFilter.vgrism: "ugv",
    SwiftFilter.ugrism: "UGRISM",
    # SwiftFilter.magnifier: "umg",
    # SwiftFilter.blocked: "ubl",
    # SwiftFilter.unknown: "uun",
}


def filter_to_obs_string(filter_type: SwiftFilter) -> str:
    """description of how the FITS file headers denote which filter was used for taking the image"""

    return filter_to_obs_string_dict[filter_type]


def obs_string_to_filter(filter_str: str) -> SwiftFilter:
    inverse_dict = {v: k for k, v in filter_to_obs_string_dict.items()}
    return inverse_dict[filter_str]


class SwiftUVOTImageType(str, Enum):
    raw = "rw"
    detector = "dt"
    sky_units = "sk"
    exposure_map = "ex"

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]


def swift_orbit_id_from_obsid(obsid: SwiftObservationID) -> SwiftOrbitID:
    obsid_int = int(obsid)
    orbit_int = round(obsid_int / 1000)
    return SwiftOrbitID(f"{orbit_int:08}")


def swift_observation_id_from_int(number: int) -> Optional[SwiftObservationID]:
    converted_string = f"{number:011}"
    if len(converted_string) != 11:
        return None
    return SwiftObservationID(converted_string)


def swift_orbit_id_from_int(number: int) -> Optional[SwiftOrbitID]:
    converted_string = f"{number:08}"
    if len(converted_string) != 8:
        return None
    return SwiftOrbitID(converted_string)


@dataclass
class PixelCoord:
    """Use floats instead of ints to allow sub-pixel addressing"""

    x: float
    y: float


@dataclass
class SwiftStackedUVOTImage:
    stacked_image: SwiftUVOTImage
    # Tuple with the obsids, filenames, and extensions of each image that contributed to the stacked image
    sources: List[Tuple[SwiftObservationID, pathlib.Path, int]]
    # sum of exposure times of the stacked images
    exposure_time: float
    filter_type: SwiftFilter
