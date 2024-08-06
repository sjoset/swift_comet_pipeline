import pathlib
import numpy as np
import os
import glob

from astropy.io import fits
from astropy.wcs import WCS

from typing import TypeAlias, List
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage, SwiftUVOTImageType


SwiftObservationID: TypeAlias = str
SwiftOrbitID: TypeAlias = str


def swift_orbit_id_from_obsid(obsid: SwiftObservationID) -> SwiftOrbitID:
    obsid_int = int(obsid)
    orbit_int = round(obsid_int / 1000)
    return SwiftOrbitID(f"{orbit_int:08}")


def swift_observation_id_from_int(number: int) -> SwiftObservationID | None:
    converted_string = f"{number:011}"
    if len(converted_string) != 11:
        return None
    return SwiftObservationID(converted_string)


def swift_orbit_id_from_int(number: int) -> SwiftOrbitID | None:
    converted_string = f"{number:08}"
    if len(converted_string) != 8:
        return None
    return SwiftOrbitID(converted_string)


class SwiftData:
    """
    Class that takes a directory that points to Swift data, which is assumed to be in this format:
        data_path/
            [observation id]/
                uvot/
                    image/
                        sw[observation id][filter]_[image type].img.gz

    Then given an observation id, filter, and image type, we can construct the path to this file
    """

    def __init__(self, data_path: pathlib.Path):
        self.base_path = data_path

    def get_all_observation_ids(self) -> List[SwiftObservationID]:
        """
        build a list of folders in the swift data directory, filtering any directories
        that don't match the naming structure of 11 numerical digits, and returns a list of every observation id found
        """
        # TODO: don't assume the path we're handed is valid and that it exists!

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
    ) -> List[pathlib.Path] | None:
        """
        Given an observation ID, filter type, and image type, returns a list of FITS files that match.
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

    def get_uvot_image(
        self, obsid: SwiftObservationID, fits_filename: str, fits_extension: int
    ) -> SwiftUVOTImage:
        image_path = self.get_uvot_image_directory(obsid=obsid) / pathlib.Path(
            fits_filename
        )
        image_data: SwiftUVOTImage = fits.getdata(image_path, ext=fits_extension)  # type: ignore
        return image_data

    def get_uvot_image_wcs(
        self, obsid: SwiftObservationID, fits_filename: str, fits_extension: int
    ) -> WCS:
        image_path = self.get_uvot_image_directory(obsid=obsid) / pathlib.Path(
            fits_filename
        )

        with fits.open(image_path) as hdul:
            header = hdul[fits_extension].header  # type: ignore
            wcs = WCS(header)

        return wcs
