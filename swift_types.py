import pathlib
import numpy as np
import pandas as pd
import os
import glob

from astropy.time import Time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, TypeAlias, List, Tuple


__all__ = [
    "SwiftData",
    "SwiftObservationLog",
    "SwiftUVOTImage",
    "SwiftStackedUVOTImage",
    "SwiftStackingMethod",
    "SwiftFilter",
    "SwiftUVOTImageType",
    "SwiftOrbitID",
    "SwiftObservationID",
    "swift_orbit_id_from_obsid",
    "swift_observation_id_from_int",
    "PixelCoord",
    "filter_to_string",
    "filter_to_file_string",
    "file_string_to_filter",
    "filter_to_obs_string",
    "obs_string_to_filter",
]


SwiftObservationLog: TypeAlias = pd.DataFrame
SwiftUVOTImage: TypeAlias = np.ndarray
SwiftObservationID: TypeAlias = str
SwiftOrbitID: TypeAlias = str


class SwiftStackingMethod(str, Enum):
    summation = "sum"
    median = "median"

    @classmethod
    def all_stacking_methods(cls):
        return [x for x in cls]


class SwiftPixelResolution(str, Enum):
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


def filter_to_string(filter_type: SwiftFilter) -> str:
    return filter_to_file_string(filter_type)


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
    coincidence_corrected: bool
    # TODO: The detector scale might be in the FITS source file headers
    detector_scale: SwiftPixelResolution
    stacking_method: SwiftStackingMethod
    observation_mid_time: str


class SwiftData:
    """
    Class that takes a directory that points to Swift data, which is assumed to be in this format:
        data_path/
            [observation id]/
                uvot/
                    image/
                        sw[observation id][filter]_[image type].img.gz

    Then given the observation id, filter, and image type, we can construct the path to this file
    """

    def __init__(self, data_path: pathlib.Path):
        self.base_path = data_path

    def get_all_observation_ids(self) -> List[SwiftObservationID]:
        """
        build a list of folders in the swift data directory, filtering any directories
        that don't match the naming structure of 11 numerical digits, and returns a list of every observation id found
        """
        # get a list of everything in the top-level data directory
        file_list = os.listdir(self.base_path)

        # take observation IDs, combine them with the path to the data to get full paths to everything in our swift data folder
        file_paths = list(map(lambda x: pathlib.Path(self.base_path / x), file_list))

        # filter out non-directories
        dir_paths = [dirname for dirname in filter(lambda x: x.is_dir(), file_paths)]

        # valid obsids should be 11 characters long: remove anything that is not 11 characters
        correct_length_names = [
            dirname for dirname in filter(lambda x: len(x.name) == 11, dir_paths)
        ]

        # keep only the numeric names like '00020405001'
        numeric_names = [
            dirname
            for dirname in filter(lambda x: x.name.isnumeric(), correct_length_names)
        ]

        return list(map(lambda x: SwiftObservationID(x.name), numeric_names))

    def get_all_orbit_ids(self) -> List[SwiftOrbitID]:
        """
        build a list of orbit ids based on the folder names in the swift data directory
        """

        # these should already be validated, so we can just chop off the last three digits to get the orbit id
        obsid_list = self.get_all_observation_ids()

        return np.unique(list(map(swift_orbit_id_from_obsid, obsid_list)))  # type: ignore

    def get_swift_uvot_image_paths(
        self,
        obsid: SwiftObservationID,
        filter_type: SwiftFilter,
        image_type: SwiftUVOTImageType = SwiftUVOTImageType.sky_units,
    ) -> Optional[List[pathlib.Path]]:
        """
        Given an observation ID, filter type, and image type, returns a list of FITS files that match
        Some observations have multiple files using the same filter, so we have to do it this way
        """
        filter_string = filter_to_file_string(filter_type)

        # TODO: find a directory where there are multiple _sk.img.gz files so we can make sure this is the proper way to handle this
        image_path = self.get_uvot_image_directory(obsid)
        image_name_base = "sw" + obsid + filter_string + "_" + image_type
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.img.gz")

        if len(matching_files) == 0:
            return None

        return list(map(pathlib.Path, matching_files))

    def get_uvot_image_directory(self, obsid: SwiftObservationID) -> pathlib.Path:
        """Returns a path to the directory containing the uvot images of the given observation id"""
        image_path = self.base_path / obsid / "uvot" / "image"
        return image_path
