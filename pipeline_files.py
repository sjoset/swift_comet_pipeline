import calendar
import pathlib
import glob
from typing import List, Optional

import numpy as np

from astropy.time import Time

from epochs import Epoch
from swift_filter import filter_to_file_string, SwiftFilter
from stacking import StackingMethod


# __all__ = [
#     "epoch_filenames_from_epoch_list",
#     "epoch_name_from_epoch_path",
#     "stacked_fits_path",
#     "default_epoch_dir",
#     "default_analysis_dir",
#     "analysis_background_path",
#     "analysis_bg_subtracted_path",
#     "analysis_qh2o_vs_aperture_radius_path",
# ]
__all__ = ["PipelineFiles"]


"""
Folder structure, starting at product_save_path

Fixed:
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


Variable:
    epoch_dir_path/
        {epoch_name}.parquet
        ...

    stack_dir_path/
        {epoch_name}.parquet   (vetoed observations omitted from this database)
        {epoch_name}_{filter}_{stack type}.fits

    # TODO: name these files
    analysis_path/
        {epoch_name}/
            background: size and positions of both apertures, values found
            bg_subtracted_uw1.fits:
            bg_subtracted_uvv.fits: fits files after background subtraction, with the same fits header information from the stacked image that produced them
            qh2o_vs_aperture_radius.csv : vary circular aperture radius up to the maximum (ask for input of min_r and max_r) and compute qh2o
            qh2o_vs_from_profile_gaussian_fit : vary angular cut of comet profile, estimate radius, and compute qh2o as a function
                of profile cut angle (estimation using gaussian) and maybe cutoff - 3, 4, 5 sigma
            qh2o_vs_from_profile : vary angular cut of comet profile, estimate radius, and compute qh2o as a function of profile cut angle
"""


class PipelineFiles:
    def __init__(self, product_save_path: pathlib.Path, expect_epochs=False):
        self.product_save_path = product_save_path
        self.epoch_dir_path = product_save_path / pathlib.Path("epochs")
        self.stack_dir_path = product_save_path / pathlib.Path("stacked")
        self.analysis_base_path = product_save_path / pathlib.Path("analysis")

        # list of epoch files
        self.epoch_file_paths = []
        if expect_epochs:
            epoch_file_paths = self._get_epoch_file_paths()
            if epoch_file_paths is None:
                print(
                    "No epoch files were found, but expected at this point in the pipeline! Exiting."
                )
                exit(1)
            self.epoch_file_paths = epoch_file_paths

        # list of processed epoch files associated with a stacked image
        stacked_epoch_paths = [
            self.get_stacked_epoch_path(x) for x in self.epoch_file_paths
        ]
        # filter out any that don't exist because the epoch hasn't been stacked yet
        self.stacked_epoch_paths = list(
            filter(lambda x: x.is_file(), stacked_epoch_paths)
        )

    def get_observation_log_path(self):
        return self.product_save_path / pathlib.Path("observation_log.parquet")

    def get_comet_orbital_data_path(self):
        return (
            self.product_save_path
            / pathlib.Path("orbital_data")
            / pathlib.Path("horizons_comet_orbital_data.csv")
        )

    def get_earth_orbital_data_path(self):
        return (
            self.product_save_path
            / pathlib.Path("orbital_data")
            / pathlib.Path("horizons_earth_orbital_data.csv")
        )

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

    def _get_epoch_file_paths(self) -> Optional[List[pathlib.Path]]:
        """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
        glob_pattern = str(self.epoch_dir_path / pathlib.Path("*.parquet"))
        epoch_filename_list = sorted(glob.glob(glob_pattern))
        if len(epoch_filename_list) == 0:
            return None
        return [pathlib.Path(x) for x in epoch_filename_list]

    def get_stacked_epoch_path(self, epoch_path) -> pathlib.Path:
        """Epoch saved after being filtered for veto etc. are stored with the same name, but in the stack folder"""
        return self.stack_dir_path / epoch_path.name

    def get_stacked_image_path(
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

    def get_analysis_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
        """Naming convention for the analysis products: each epoch gets its own folder to store results"""
        return self.analysis_base_path / epoch_path.stem

    def get_analysis_background_path(self, epoch_path: pathlib.Path) -> pathlib.Path:
        """Naming convention for bacground analysis summary"""
        return self.get_analysis_path(epoch_path) / pathlib.Path(
            "background_analysis.yaml"
        )

    def get_analysis_bg_subtracted_path(
        self, epoch_path: pathlib.Path, filter_type: SwiftFilter
    ) -> pathlib.Path:
        """Naming convention for background-subtracted fits images, for later inspection to see if a sane background was determined"""
        return self.get_analysis_path(epoch_path) / pathlib.Path(
            f"bg_subtracted_{filter_to_file_string(filter_type)}.fits"
        )

    def get_analysis_qh2o_vs_aperture_radius_path(
        self, epoch_path: pathlib.Path
    ) -> pathlib.Path:
        """Naming convention for calculated water production versus circular aperture radius"""
        return self.get_analysis_path(epoch_path) / pathlib.Path(
            "qh2o_vs_aperture_radius.csv"
        )


# def default_epoch_dir(product_save_path: pathlib.Path) -> pathlib.Path:
#     return product_save_path / "epochs"
#
#
# def epoch_filenames_from_epoch_list(epoch_list: List[Epoch]) -> List[pathlib.Path]:
#     epoch_path_list = []
#     for i, epoch in enumerate(epoch_list):
#         epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
#         day = epoch_start.day
#         month = calendar.month_abbr[epoch_start.month]
#         year = epoch_start.year
#
#         filename = f"{i:03d}_{year}_{day:02d}_{month}.parquet"
#         epoch_path_list.append(filename)
#
#     return epoch_path_list
#
#
# def epoch_name_from_epoch_path(epoch_path: pathlib.Path) -> str:
#     return str(epoch_path.stem)
#
#
# def stacked_fits_path(
#     stack_dir_path: pathlib.Path,
#     epoch_path: pathlib.Path,
#     filter_type: SwiftFilter,
#     stacking_method: StackingMethod,
# ) -> pathlib.Path:
#     epoch_name = epoch_name_from_epoch_path(epoch_path=epoch_path)
#     filter_string = filter_to_file_string(filter_type)
#     fits_filename = f"{epoch_name}_{filter_string}_{stacking_method}.fits"
#
#     return stack_dir_path / pathlib.Path(fits_filename)
#
#
# def default_analysis_dir(
#     product_save_path: pathlib.Path, epoch_name: str
# ) -> pathlib.Path:
#     return product_save_path / pathlib.Path("analysis") / pathlib.Path(epoch_name)
#
#
# def analysis_background_path(analysis_dir: pathlib.Path) -> pathlib.Path:
#     return analysis_dir / pathlib.Path("background_analysis.yaml")
#
#
# def analysis_bg_subtracted_path(
#     analysis_dir: pathlib.Path, filter_type: SwiftFilter
# ) -> pathlib.Path:
#     return analysis_dir / pathlib.Path(
#         f"bg_subtracted_{filter_to_file_string(filter_type)}.fits"
#     )
#
#
# def analysis_qh2o_vs_aperture_radius_path(analysis_dir: pathlib.Path) -> pathlib.Path:
#     return analysis_dir / pathlib.Path("qh2o_vs_aperture_radius.csv")
