from itertools import product
import pathlib
import numpy as np
import os
import glob
from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS

from typing import List
from swift_comet_pipeline.swift.swift_filter_to_string import filter_to_file_string
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_ids import SwiftObservationID, SwiftOrbitID
from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage
from swift_comet_pipeline.types.swift_uvot_image_type import SwiftUVOTImageType


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


@dataclass
class SwiftFITSObservation:
    orbit_id: SwiftOrbitID
    observation_id: SwiftObservationID
    path: pathlib.Path
    observation_mode: SwiftImageMode


class SwiftData:
    """
    Class that takes a directory that points to Swift data, which is assumed to be in this format:
        data_path/
            [observation id]/
                uvot/
                    image/
                        sw[observation id][filter]_[image type].img.gz
                    event/
                        sw[observation id][filter]*.evt.gz

    Then given an observation id, filter, and image type, we can construct the path to this file
    """

    def __init__(self, data_path: pathlib.Path):
        self.base_path = data_path

        self.observation_ids = self._get_all_observation_ids()
        self.orbit_ids = self._get_all_orbit_ids()

        if self.observation_ids is None:
            return

        self.observations = {}
        for obsid, ft in product(self.observation_ids, SwiftFilter.all_filters()):
            self.observations = self._get_swift_uvot_observations(
                obsid=obsid, filter_type=ft
            )

    def _get_all_observation_ids(self) -> List[SwiftObservationID] | None:
        """
        build a list of folders in the swift data directory, filtering any directories
        that don't match the naming structure of 11 numerical digits, and returns a list of every observation id found
        """
        if not self.base_path.exists():
            return None

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

    def _get_all_orbit_ids(self) -> List[SwiftOrbitID] | None:
        """
        build a list of orbit ids based on the folder names in the swift data directory
        """

        if self.observation_ids is None:
            return None

        # these should already be validated, so we can just chop off the last three digits to get the orbit id
        return np.unique(list(map(swift_orbit_id_from_obsid, self.observation_ids)))  # type: ignore

    def _get_swift_uvot_event_mode_fits_observations(
        self,
        obsid: SwiftObservationID,
        filter_type: SwiftFilter,
    ) -> List[SwiftFITSObservation] | None:
        """
        Given an observation ID, filter type, and image type, returns a list of event-mode FITS files that match.
        """
        filter_string = filter_to_file_string(filter_type)

        # TODO: find a directory where there are multiple _sk.img.gz files so we can make sure this is the proper way to handle this
        image_path = self.get_uvot_image_directory(
            obsid, data_mode=SwiftImageMode.event_mode
        )

        image_name_base = "sw" + obsid + filter_string
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.evt.gz")

        if len(matching_files) == 0:
            return None

        # return list(map(pathlib.Path, matching_files))
        img_paths = list(map(pathlib.Path, matching_files))
        return [
            SwiftFITSObservation(
                orbit_id=swift_orbit_id_from_obsid(obsid),
                observation_id=obsid,
                path=x,
                observation_mode=SwiftImageMode.event_mode,
            )
            for x in img_paths
        ]

    def _get_swift_uvot_data_mode_fits_observations(
        self,
        obsid: SwiftObservationID,
        filter_type: SwiftFilter,
        image_type: SwiftUVOTImageType = SwiftUVOTImageType.sky_units,
    ) -> List[SwiftFITSObservation] | None:
        """
        Given an observation ID, filter type, and image type, returns a list of FITS files that match.
        Some observations have multiple files using the same filter, so we have to do it this way
        """
        filter_string = filter_to_file_string(filter_type)

        # TODO: find a directory where there are multiple _sk.img.gz files so we can make sure this is the proper way to handle this
        image_path = self.get_uvot_image_directory(
            obsid, data_mode=SwiftImageMode.data_mode
        )
        image_name_base = "sw" + obsid + filter_string + "_" + image_type
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.img.gz")

        if len(matching_files) == 0:
            return None

        img_paths = list(map(pathlib.Path, matching_files))
        return [
            SwiftFITSObservation(
                orbit_id=swift_orbit_id_from_obsid(obsid),
                observation_id=obsid,
                path=x,
                observation_mode=SwiftImageMode.data_mode,
            )
            for x in img_paths
        ]

    def _get_swift_uvot_observations(
        self, obsid: SwiftObservationID, filter_type: SwiftFilter
    ) -> List[SwiftFITSObservation] | None:
        """
        Given an observation ID and filter type, returns a list of FITS files that match.
        Some observations have multiple files using the same filter, so we have to do it this way

        For observations that have a valid event mode file, skip the data mode image that Swift provides
        TODO: check if there are ever event mode and data mode exposures under the same obsid
        """

        event_mode_path_list = self._get_swift_uvot_event_mode_fits_observations(
            obsid=obsid, filter_type=filter_type
        )
        if event_mode_path_list is not None:
            return event_mode_path_list

        data_mode_path_list = self._get_swift_uvot_data_mode_fits_observations(
            obsid=obsid, filter_type=filter_type
        )

        # result_list = (data_mode_path_list or []) + (event_mode_path_list or [])
        # if len(result_list) == 0:
        #     result_list = None

        return data_mode_path_list

    def get_uvot_image_directory(
        self, obsid: SwiftObservationID, data_mode: SwiftImageMode
    ) -> pathlib.Path:
        """Returns a path to the directory containing the uvot images of the given observation id"""
        if data_mode == SwiftImageMode.data_mode:
            image_path = self.base_path / obsid / "uvot" / "image"
        elif data_mode == SwiftImageMode.event_mode:
            image_path = self.base_path / obsid / "uvot" / "event"

        return image_path

    def get_uvot_image(
        self,
        obsid: SwiftObservationID,
        fits_filename: str,
        fits_extension: int,
        data_mode: SwiftImageMode,
    ) -> SwiftUVOTImage:
        image_path = self.get_uvot_image_directory(
            obsid=obsid, data_mode=data_mode
        ) / pathlib.Path(fits_filename)
        image_data: SwiftUVOTImage = fits.getdata(image_path, ext=fits_extension)  # type: ignore
        return image_data

    def get_uvot_image_wcs(
        self,
        obsid: SwiftObservationID,
        fits_filename: str,
        fits_extension: int,
        data_mode: SwiftImageMode,
    ) -> WCS:
        image_path = self.get_uvot_image_directory(
            obsid=obsid, data_mode=data_mode
        ) / pathlib.Path(fits_filename)

        with fits.open(image_path) as hdul:
            header = hdul[fits_extension].header  # type: ignore
            wcs = WCS(header)

        return wcs
