import calendar
import pathlib
import glob
import yaml

from typing import List, Optional, Any
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.io import fits

from epochs import Epoch, read_epoch, write_epoch
from observation_log import read_observation_log, write_observation_log
from swift_filter import filter_to_file_string, SwiftFilter
from stacking import StackingMethod


__all__ = ["PipelineFiles"]


# TODO: update this documentation once things are settled
"""
Folder structure, starting at product_save_path

centers/
    uvv/
        {obsid}_{fits extension}.png
        ...
    uw1/
        {obsid}_{fits extension}.png
        ...

orbit/
        {comet_orbital_data_path}
        {earth_orbital_data_path}

epoch_dir_path/
    {epoch_name}.parquet
    ...

stack_dir_path/
    {epoch_name}.parquet   (vetoed observations omitted from this database)
    {epoch_name}_{filter}_{stack type}.fits

# TODO: name these files
analysis_path/
    {epoch_name}/
        background_analysis.yaml: values of background per filter with additional info on method used
        bg_subtracted_uw1.fits:
        bg_subtracted_uvv.fits: fits files after background subtraction, with the same fits header information from the stacked image that produced them
        qh2o_vs_aperture_radius.csv : vary circular aperture radius up to the maximum (ask for input of min_r and max_r) and compute qh2o
        qh2o_vs_from_profile_gaussian_fit : vary angular cut of comet profile, estimate radius, and compute qh2o as a function
            of profile cut angle (estimation using gaussian) and maybe cutoff - 3, 4, 5 sigma
        qh2o_vs_from_profile : vary angular cut of comet profile, estimate radius, and compute qh2o as a function of profile cut angle
"""


class PipelineProduct(ABC):
    def __init__(self, product_path: pathlib.Path):
        self.product_path = product_path
        self._data_product: Any = None

    @abstractmethod
    def save_product(self) -> None:
        if self._data_product is None:
            print(
                f"Attempted to save PipelineProduct {self.product_path} with no data given!"
            )

    @abstractmethod
    def load_product(self) -> None:
        pass

    @property
    def data_product(self) -> Any:
        """The data_product property."""
        return self._data_product

    @data_product.setter
    def data_product(self, data_product: Any):
        self._data_product = data_product

    def exists(self) -> bool:
        return self.product_path.exists()


class ObservationLogProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        write_observation_log(
            obs_log=self._data_product, obs_log_path=self.product_path
        )

    def load_product(self) -> None:
        self._data_product = read_observation_log(self.product_path)


class EpochProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        write_epoch(epoch=self._data_product, epoch_path=self.product_path)

    def load_product(self) -> None:
        self._data_product = read_epoch(epoch_path=self.product_path)


class StackedEpochProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        write_epoch(epoch=self._data_product, epoch_path=self.product_path)

    def load_product(self) -> None:
        self._data_product = read_epoch(epoch_path=self.product_path)


# instead of fighting fits file formatting and HDU lists, we assert that hdu[0] is an empty primary HDU
# and that hdu[1] is an ImageHDU, with relevant header and image data, for our stacked images
class StackedFitsImageProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        self._data_product.writeto(self.product_path, overwrite=True)

    def load_product(self) -> None:
        hdul = fits.open(self.product_path, lazy_load_hdus=False, memmap=True)
        self._data_product = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()


class CSVDataProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        self._data_product.to_csv(self.product_path)

    def load_product(self) -> None:
        self._data_product = pd.read_csv(self.product_path)


class YamlProduct(PipelineProduct):
    def save_product(self) -> None:
        super().save_product()
        with open(self.product_path, "w") as stream:
            try:
                yaml.safe_dump(self._data_product, stream)
            except yaml.YAMLError as exc:
                print(exc)

    def load_product(self) -> None:
        with open(self.product_path, "r") as stream:
            try:
                self._data_product = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


class PipelineFiles:
    def __init__(self, base_product_save_path: pathlib.Path):
        """Construct file paths to pipeline products for later loading/saving"""
        self.base_product_save_path = base_product_save_path
        self.orbital_data_dir_path = base_product_save_path / pathlib.Path(
            "orbital_data"
        )
        self.epoch_dir_path = base_product_save_path / pathlib.Path("epochs")
        self.stack_dir_path = base_product_save_path / pathlib.Path("stacked")
        self.analysis_base_path = base_product_save_path / pathlib.Path("analysis")

        # create these directories
        for p in [
            self.orbital_data_dir_path,
            self.epoch_dir_path,
            self.stack_dir_path,
            self.analysis_base_path,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        """Observation log and orbital info do not rely on cutting the data into epochs"""
        self.observation_log = ObservationLogProduct(
            product_path=self.base_product_save_path
            / pathlib.Path("observation_log.parquet")
        )
        self.comet_orbital_data = CSVDataProduct(
            product_path=self.orbital_data_dir_path
            / pathlib.Path("horizons_comet_orbital_data.csv"),
        )
        self.earth_orbital_data = CSVDataProduct(
            product_path=self.orbital_data_dir_path
            / pathlib.Path("horizons_earth_orbital_data.csv")
        )

        """Initialize all products as None and fill in later if epochs have been generated from the observation log"""
        # epochs
        self.epoch_products: Optional[List[EpochProduct]] = None
        # stacked epoch files that only include non-vetoed / excluded data from the original epoch that were used to generate a stacked image
        self.stacked_epoch_products: Optional[
            dict[pathlib.Path, StackedEpochProduct]
        ] = None
        # dictionary to look up stacked image by stacked epoch file path, filter, and stacking method
        self.stacked_image_products: Optional[
            dict[
                tuple[pathlib.Path, SwiftFilter, StackingMethod],
                StackedFitsImageProduct,
            ]
        ] = None
        # background analysis
        self.analysis_background_products: Optional[
            dict[pathlib.Path, YamlProduct]
        ] = None
        self.analysis_bg_subtracted_images: Optional[
            dict[
                tuple[pathlib.Path, SwiftFilter, StackingMethod],
                StackedFitsImageProduct,
            ]
        ] = None
        # Q(H2O) vs aperture radius
        self.analysis_qh2o_products: Optional[dict[pathlib.Path, CSVDataProduct]] = None

        # epochs
        self.epoch_file_paths = self._find_epoch_file_paths()
        if self.epoch_file_paths is None:
            # print("No epoch files generated")
            return
        self.epoch_products = [EpochProduct(x) for x in self.epoch_file_paths]

        # stacked epochs: given an epoch, store a corresponding file with the stacked epoch data
        stacked_epoch_dict = {}
        for epoch_path in self.epoch_file_paths:
            stacked_epoch_dict[epoch_path] = StackedEpochProduct(
                self._get_stacked_epoch_path(epoch_path=epoch_path)
            )
        self.stacked_epoch_products = stacked_epoch_dict

        # stacked images: given an epoch path, filter type, and stacking method, store a corresponding FitsImageProduct
        stacked_image_dict = {}
        for epoch_path, filter_type, stacking_method in product(
            self.epoch_file_paths,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            stacked_image_dict[
                epoch_path, filter_type, stacking_method
            ] = StackedFitsImageProduct(
                product_path=self.stack_dir_path
                / self._get_stacked_image_path(
                    epoch_path=epoch_path,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )
        self.stacked_image_products = stacked_image_dict

        # background analysis: given an epoch path, store a corresponding yaml file containing relevant background analysis
        bg_analysis_dict = {}
        for epoch_path in self.epoch_file_paths:
            bg_analysis_dict[epoch_path] = YamlProduct(
                product_path=self._get_analysis_background_path(epoch_path=epoch_path)
            )
        self.analysis_background_products = bg_analysis_dict

        # background-subtracted images
        bg_subtracted_dict = {}
        for epoch_path, filter_type, stacking_method in product(
            self.epoch_file_paths,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            bg_subtracted_dict[
                epoch_path, filter_type, stacking_method
            ] = StackedFitsImageProduct(
                product_path=self._get_analysis_bg_subtracted_fits_path(
                    epoch_path=epoch_path,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )
        self.analysis_bg_subtracted_images = bg_subtracted_dict

        # Water production versus aperture radius
        qh2o_dict = {}
        for epoch_path in self.epoch_file_paths:
            qh2o_dict[epoch_path] = CSVDataProduct(
                product_path=self._get_analysis_qh2o_vs_aperture_radius_path(
                    epoch_path=epoch_path
                )
            )
        self.analysis_qh2o_products = qh2o_dict

    def determine_epoch_file_paths(self, epoch_list: List[Epoch]) -> List[pathlib.Path]:
        """Controls naming of the epoch files generated after slicing the original observation log into time windows, naming based on the MID_TIME of observation"""
        epoch_path_list = []
        for i, epoch in enumerate(epoch_list):
            epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
            day = epoch_start.day
            month = calendar.month_abbr[epoch_start.month]
            year = epoch_start.year

            filename = pathlib.Path(f"{i:03d}_{year}_{day:02d}_{month}.parquet")
            epoch_path_list.append(self.epoch_dir_path / filename)

        return epoch_path_list

    def _find_epoch_file_paths(self) -> Optional[List[pathlib.Path]]:
        """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
        glob_pattern = str(self.epoch_dir_path / pathlib.Path("*.parquet"))
        epoch_filename_list = sorted(glob.glob(glob_pattern))
        if len(epoch_filename_list) == 0:
            return None
        return [pathlib.Path(x) for x in epoch_filename_list]

    def _get_stacked_epoch_path(self, epoch_path) -> pathlib.Path:
        """Epoch saved after being filtered for veto etc. are stored with the same name, but in the stack folder"""
        return self.stack_dir_path / epoch_path.name

    def _get_stacked_image_path(
        self,
        epoch_path: pathlib.Path,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> pathlib.Path:
        """Naming convention given the epoch path, filter, and stacking method to use for the stacked FITS files"""
        epoch_name = epoch_path.stem
        filter_string = filter_to_file_string(filter_type)
        fits_filename = f"{epoch_name}_{filter_string}_{stacking_method}.fits"

        return self.stack_dir_path / pathlib.Path(fits_filename)

    def _get_analysis_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
        """Naming convention for the analysis products: each epoch gets its own folder to store results"""
        return self.analysis_base_path / epoch_path.stem

    def _get_analysis_background_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
        """Naming convention for bacground analysis summary"""
        return self._get_analysis_path(epoch_path) / pathlib.Path(
            "background_analysis.yaml"
        )

    def _get_analysis_bg_subtracted_fits_path(
        self,
        epoch_path: pathlib.Path,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> pathlib.Path:
        """Naming convention for background-subtracted fits images, for later inspection to see if a sane background was determined"""
        return self._get_analysis_path(epoch_path) / pathlib.Path(
            f"bg_subtracted_{filter_to_file_string(filter_type)}_{stacking_method}.fits"
        )

    def _get_analysis_qh2o_vs_aperture_radius_path(
        self, epoch_path: pathlib.Path
    ) -> pathlib.Path:
        """Naming convention for calculated water production versus circular aperture radius"""
        return self._get_analysis_path(epoch_path) / pathlib.Path(
            "qh2o_vs_aperture_radius.csv"
        )


# class PipelineFiles:
#     def __init__(self, product_save_path: pathlib.Path, expect_epochs=False):
#         self.product_save_path = product_save_path
#         self.epoch_dir_path = product_save_path / pathlib.Path("epochs")
#         self.stack_dir_path = product_save_path / pathlib.Path("stacked")
#         self.analysis_base_path = product_save_path / pathlib.Path("analysis")
#
#         # list of epoch files
#         self.epoch_file_paths = []
#         if expect_epochs:
#             epoch_file_paths = self._get_epoch_file_paths()
#             if epoch_file_paths is None:
#                 print(
#                     "No epoch files were found, but expected at this point in the pipeline! Exiting."
#                 )
#                 exit(1)
#             self.epoch_file_paths = epoch_file_paths
#
#         # list of processed epoch files associated with a stacked image
#         stacked_epoch_paths = [
#             self.get_stacked_epoch_path(x) for x in self.epoch_file_paths
#         ]
#         # filter out any that don't exist because the epoch hasn't been stacked yet
#         self.stacked_epoch_paths = list(
#             filter(lambda x: x.is_file(), stacked_epoch_paths)
#         )
#
#     def get_observation_log_path(self):
#         return self.product_save_path / pathlib.Path("observation_log.parquet")
#
#     def get_comet_orbital_data_path(self):
#         return (
#             self.product_save_path
#             / pathlib.Path("orbital_data")
#             / pathlib.Path("horizons_comet_orbital_data.csv")
#         )
#
#     def get_earth_orbital_data_path(self):
#         return (
#             self.product_save_path
#             / pathlib.Path("orbital_data")
#             / pathlib.Path("horizons_earth_orbital_data.csv")
#         )
#
#     def determine_epoch_file_paths(self, epoch_list: List[Epoch]) -> List[pathlib.Path]:
#         """Controls naming of the epoch files generated after slicing the original observation log into time windows, naming based on the MID_TIME of observation"""
#         epoch_path_list = []
#         for i, epoch in enumerate(epoch_list):
#             epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
#             day = epoch_start.day
#             month = calendar.month_abbr[epoch_start.month]
#             year = epoch_start.year
#
#             filename = pathlib.Path(f"{i:03d}_{year}_{day:02d}_{month}.parquet")
#             epoch_path_list.append(self.epoch_dir_path / filename)
#
#         return epoch_path_list
#
#     def _get_epoch_file_paths(self) -> Optional[List[pathlib.Path]]:
#         """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
#         glob_pattern = str(self.epoch_dir_path / pathlib.Path("*.parquet"))
#         epoch_filename_list = sorted(glob.glob(glob_pattern))
#         if len(epoch_filename_list) == 0:
#             return None
#         return [pathlib.Path(x) for x in epoch_filename_list]
#
#     def get_stacked_epoch_path(self, epoch_path) -> pathlib.Path:
#         """Epoch saved after being filtered for veto etc. are stored with the same name, but in the stack folder"""
#         return self.stack_dir_path / epoch_path.name
#
#     def get_stacked_image_path(
#         self,
#         epoch_path: pathlib.Path,
#         filter_type: SwiftFilter,
#         stacking_method: StackingMethod,
#     ) -> pathlib.Path:
#         """Naming convention given the epoch path, filter, and stacking method to use for the stacked FITS files"""
#         epoch_name = epoch_path.stem
#         filter_string = filter_to_file_string(filter_type)
#         fits_filename = f"{epoch_name}_{filter_string}_{stacking_method}.fits"
#
#         return self.stack_dir_path / pathlib.Path(fits_filename)
#
#     def get_analysis_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
#         """Naming convention for the analysis products: each epoch gets its own folder to store results"""
#         return self.analysis_base_path / epoch_path.stem
#
#     def get_analysis_background_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
#         """Naming convention for bacground analysis summary"""
#         return self.get_analysis_path(epoch_path) / pathlib.Path(
#             "background_analysis.yaml"
#         )
#
#     def get_analysis_bg_subtracted_path(
#         self, epoch_path: pathlib.Path, filter_type: SwiftFilter
#     ) -> pathlib.Path:
#         """Naming convention for background-subtracted fits images, for later inspection to see if a sane background was determined"""
#         return self.get_analysis_path(epoch_path) / pathlib.Path(
#             f"bg_subtracted_{filter_to_file_string(filter_type)}.fits"
#         )
#
#     def get_analysis_qh2o_vs_aperture_radius_path(
#         self, epoch_path: pathlib.Path
#     ) -> pathlib.Path:
#         """Naming convention for calculated water production versus circular aperture radius"""
#         return self.get_analysis_path(epoch_path) / pathlib.Path(
#             "qh2o_vs_aperture_radius.csv"
#         )
