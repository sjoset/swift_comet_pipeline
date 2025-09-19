import pathlib
import numpy as np
import os
import glob
from functools import cache
from itertools import product

from astropy.io import fits

from swift_comet_pipeline.scp_types import *
from swift_comet_pipeline.scp_types.compound.swift_dataset import SwiftDataset
from swift_comet_pipeline.scp_types.compound.swift_level_2_fits import (
    SwiftLevel2FITSObservation,
)
from swift_comet_pipeline.swift.filters.uvot_filter_to_string import (
    filter_to_file_string,
)
from swift_comet_pipeline.swift.swift_orbit_id import swift_orbit_id_from_obsid


class SwiftData(SwiftDataset):
    """
    Class that takes a directory that points to Swift data, which is assumed to be in this format:
        data_path/
            [observation id]/
                uvot/
                    image/
                        sw[observation id][filter]_[image type].img.gz
                    event/
                        sw[observation id][filter]*.evt.gz

    Gathers all available observations into self.observations
    """

    def __init__(self, data_path: pathlib.Path):
        self.base_path = data_path

        self.observation_ids = self._find_all_observation_ids()
        self.orbit_ids = self._get_all_orbit_ids()
        self.observations: dict[
            tuple[SwiftObservationID, UvotFilter],
            list[SwiftLevel2FITSObservation] | None,
        ] = {}

        if self.observation_ids is None:
            return

        for obsid, ft in product(self.observation_ids, UvotFilter.all_filters()):
            self.observations[obsid, ft] = self.get_swift_uvot_observations(
                obsid=obsid, filter_type=ft
            )

    def _find_all_observation_ids(self) -> list[SwiftObservationID] | None:
        """
        build a list of folders in the swift data directory, filtering any directories
        that don't match the naming structure of 11 numerical digits, and returns a list of every observation id found
        """
        if not self.base_path.exists():
            print(f"Base path {self.base_path} doesnt exist!")
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

    def _get_all_orbit_ids(self) -> list[SwiftOrbitID] | None:
        """
        build a list of orbit ids based on the folder names in the swift data directory
        """

        if self.observation_ids is None:
            return None

        # these should already be validated, so we can just chop off the last three digits to get the orbit id
        return np.unique(list(map(swift_orbit_id_from_obsid, self.observation_ids)))  # type: ignore

    def get_swift_uvot_event_mode_fits_observations(
        self,
        obsid: SwiftObservationID,
        filter_type: UvotFilter,
    ) -> list[SwiftLevel2FITSObservation] | None:
        """
        Given an observation ID, filter type, and image type, returns a list of event-mode FITS files that match.
        """
        filter_string = filter_to_file_string(filter_type)

        image_path = self.get_observation_image_directory(
            obsid, image_mode=UvotImageMode.event_mode
        )

        image_name_base = "sw" + obsid + filter_string
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.evt.gz")

        if len(matching_files) == 0:
            return None

        img_paths = list(map(pathlib.Path, matching_files))
        return [
            SwiftLevel2FITSObservation(
                orbit_id=swift_orbit_id_from_obsid(obsid),
                observation_id=obsid,
                fits_path=x,
                observation_mode=UvotImageMode.event_mode,
                filter_type=filter_type,
            )
            for x in img_paths
        ]

    def get_swift_uvot_data_mode_fits_observations(
        self,
        obsid: SwiftObservationID,
        filter_type: UvotFilter,
        image_type: SwiftUvotImageType = SwiftUvotImageType.sky_units,
    ) -> list[SwiftLevel2FITSObservation] | None:
        """
        Given an observation ID, filter type, and image type, returns a list of FITS files that match.
        Defaults to sky images ('_sk.img.gz') only.
        Some observations have multiple FITS files using the same filter in the same folder,
        so we have to do it this way
        """
        filter_string = filter_to_file_string(filter_type)

        # TODO: find a directory where there are multiple _sk.img.gz files so we can make sure this is the proper way to handle this
        image_path = self.get_observation_image_directory(
            obsid, image_mode=UvotImageMode.data_mode
        )
        image_name_base = "sw" + obsid + filter_string + "_" + image_type
        image_name = image_path / image_name_base

        matching_files = glob.glob(str(image_name) + "*.img.gz")

        if len(matching_files) == 0:
            return None

        img_paths = list(map(pathlib.Path, matching_files))
        return [
            SwiftLevel2FITSObservation(
                orbit_id=swift_orbit_id_from_obsid(obsid),
                observation_id=obsid,
                fits_path=x,
                observation_mode=UvotImageMode.data_mode,
                filter_type=filter_type,
            )
            for x in img_paths
        ]

    def get_swift_uvot_observations(
        self, obsid: SwiftObservationID, filter_type: UvotFilter
    ) -> list[SwiftLevel2FITSObservation] | None:
        """
        Given an observation ID and filter type, returns a list of FITS files that match.
        Some observations have multiple files using the same filter, so we have to do it this way

        For observations that have a valid event mode file, skip the data mode image that Swift provides
        and only include the event mode image
        TODO: check if there are ever event mode and data mode exposures under the same obsid
        """

        event_mode_path_list = self.get_swift_uvot_event_mode_fits_observations(
            obsid=obsid, filter_type=filter_type
        )
        if event_mode_path_list is not None:
            return event_mode_path_list

        data_mode_path_list = self.get_swift_uvot_data_mode_fits_observations(
            obsid=obsid, filter_type=filter_type
        )

        return data_mode_path_list

    def get_observation_image_directory(
        self, obsid: SwiftObservationID, image_mode: UvotImageMode
    ) -> pathlib.Path:
        """Returns a path to the directory containing the uvot images of the given observation id"""
        if image_mode == UvotImageMode.data_mode:
            image_path = self.base_path / pathlib.Path(obsid) / "uvot" / "image"
        elif image_mode == UvotImageMode.event_mode:
            image_path = self.base_path / pathlib.Path(obsid) / "uvot" / "event"

        return image_path

    @cache
    def get_observation_image(
        self,
        obsid: SwiftObservationID,
        image_mode: UvotImageMode,
        fits_filename: str,
        extension_id: int,
    ) -> SwiftUvotImage | None:
        fits_path = self.get_observation_image_directory(
            obsid=obsid, image_mode=image_mode
        ) / pathlib.Path(fits_filename)

        if image_mode == UvotImageMode.data_mode:
            return fits.getdata(fits_path, extension_id)  # type: ignore
        elif image_mode == UvotImageMode.event_mode:
            return event_mode_fits_to_image_simple(
                fits_path=fits_path, extension_id=extension_id
            )


@cache
def event_mode_fits_to_image_simple(
    fits_path: pathlib.Path, extension_id: int
) -> SwiftUvotImage:
    ev_hdr = fits.getheader(fits_path, extension_id)
    x_min, x_max = ev_hdr["TLMIN6"], ev_hdr["TLMAX6"]
    y_min, y_max = ev_hdr["TLMIN7"], ev_hdr["TLMAX7"]
    x_size = x_max - x_min
    y_size = y_max - y_min
    ev_table = fits.getdata(fits_path, extension_id)
    assert ev_table is not None
    img, _, _ = np.histogram2d(ev_table["X"], ev_table["Y"], bins=(x_size, y_size))  # type: ignore
    img = img.T
    # binned_img = block_reduce(data=img, block_size=2)
    # return img.T
    return img


# TODO: function to go from fits_path of event mode bintable image to its sky image
# so we have the option to use those instead during the stacking phase

# def get_uvot_data_mode_image(
#     self,
#     obsid: SwiftObservationID,
#     fits_filename: str,
#     fits_extension: int,
#     # data_mode: UvotImageMode,
# ) -> SwiftUvotImage:
#     image_path = self.get_uvot_image_directory(
#         obsid=obsid, data_mode=data_mode
#     ) / pathlib.Path(fits_filename)
#     image_data: SwiftUvotImage = fits.getdata(image_path, ext=fits_extension)  # type: ignore
#     return image_data


# def get_uvot_image_wcs(
#     self,
#     obsid: SwiftObservationID,
#     fits_filename: str,
#     fits_extension: int,
#     data_mode: UvotImageMode,
# ) -> WCS:
#     image_path = self._get_observation_image_directory(
#         obsid=obsid, image_mode=data_mode
#     ) / pathlib.Path(fits_filename)
#
#     with fits.open(image_path) as hdul:
#         header = hdul[fits_extension].header  # type: ignore
#         wcs = WCS(header)
#
#     return wcs
