import os
import pathlib
import glob
import pandas as pd

from enum import Enum
from dataclasses import dataclass
from typing import Optional, TypeAlias, List


__all__ = [
    "SwiftObservationLog",
    "SwiftFilterType",
    "SwiftUVOTImageType",
    "SwiftObservationID",
    "SwiftData",
]


SwiftObservationLog: TypeAlias = pd.DataFrame


class SwiftFilterType(str, Enum):
    uuu = "uuu"
    ubb = "ubb"
    uvv = "uvv"
    uw1 = "uw1"
    uw2 = "uw2"
    um2 = "um2"
    white = "uwh"
    vgrism = "ugv"
    ugrism = "ugu"
    magnifier = "umg"
    blocked = "ubl"
    unknown = "uun"

    @classmethod
    def all_filters(cls):
        return [x for x in cls]


class SwiftUVOTImageType(str, Enum):
    raw = "rw"
    detector = "dt"
    sky_units = "sk"
    exposure_map = "ex"

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]


@dataclass
class SwiftOrbitID:
    """Swift orbit ids are 8 digit numbers assigned per Swift orbit"""

    orbit_id: str


@dataclass
class SwiftObservationID:
    """Swift observation IDs are 11 digit numbers: [orbit id] followed by 3 digit identifier"""

    obsid: str


def swift_orbit_id_from_obsid(obsid: SwiftObservationID) -> SwiftOrbitID:
    obsid_int = int(obsid.obsid)
    orbit_int = round(obsid_int / 1000)
    return SwiftOrbitID(f"{orbit_int:08}")


def swift_observation_id_from_int(number: int) -> Optional[SwiftObservationID]:
    converted_string = f"{number:011}"
    if len(converted_string) != 11:
        return None
    return SwiftObservationID(converted_string)


class SwiftData:
    """
    Takes a directory that points to Swift data, which is assumed to be in this format:
        data_path/
            [observation id]/
                uvot/
                    image/
                        sw[observation id][filter]_[image type].img.gz

    Then given the observation id, filter, and image type, we can construct the path to this file
    """

    def __init__(self, data_path: pathlib.Path):
        self.base_path = data_path

    def get_all_observation_ids_old(self) -> List[SwiftObservationID]:
        """
        build a list of folders in the swift data directory, which should all be named for their observation id
        returns a list of every observation id found
        """
        # get a list of everything in the top-level data directory
        file_list = os.listdir(self.base_path)

        return list(map(SwiftObservationID, file_list))

    def get_all_observation_ids(self) -> List[SwiftObservationID]:
        """
        build a list of folders in the swift data directory, which should all be named for their observation id
        returns a list of every observation id found
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

    def get_swift_uvot_image_paths(
        self,
        obsid: SwiftObservationID,
        filter_type: SwiftFilterType,
        image_type: SwiftUVOTImageType,
    ) -> Optional[List[pathlib.Path]]:
        """
        Given an observation ID, filter type, and image type, returns a list of FITS files that match
        Some observations have multiple files using the same filter, so we have to do it this way
        """
        # TODO: find a directory where this happens so we can make sure this is the proper way to handle this
        image_path = self.base_path / obsid.obsid / "uvot" / "image"
        image_name_base = "sw" + obsid.obsid + filter_type + "_" + image_type
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.img.gz")

        if len(matching_files) == 0:
            return None

        return list(map(pathlib.Path, matching_files))

    def swift_uvot_image_directory(self, obsid: SwiftObservationID) -> pathlib.Path:
        """Returns a path to the directory containing the uvot images of the given observation id"""
        image_path = self.base_path / obsid.obsid / "uvot" / "image"
        return image_path
