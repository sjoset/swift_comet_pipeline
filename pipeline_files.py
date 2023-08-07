import calendar
import pathlib
from typing import List

import numpy as np

from astropy.time import Time

from epochs import Epoch
from swift_filter import filter_to_file_string, SwiftFilter
from stacking import StackingMethod


__all__ = [
    "epoch_filenames_from_epoch_list",
    "epoch_name_from_epoch_path",
    "stacked_fits_path",
    "get_default_epoch_dir",
]


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
"""


# def file_name_from_epoch(epoch: Epoch) -> str:
#     epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
#     day = epoch_start.day
#     month = calendar.month_abbr[epoch_start.month]
#     year = epoch_start.year
#
#     return f"{year}_{day:02d}_{month}"


def get_default_epoch_dir(product_save_path: pathlib.Path) -> pathlib.Path:
    return product_save_path / "epochs"


def epoch_filenames_from_epoch_list(epoch_list: List[Epoch]) -> List[pathlib.Path]:
    epoch_path_list = []
    for i, epoch in enumerate(epoch_list):
        epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
        day = epoch_start.day
        month = calendar.month_abbr[epoch_start.month]
        year = epoch_start.year

        filename = f"{i:03d}_{year}_{day:02d}_{month}.parquet"
        epoch_path_list.append(filename)

    return epoch_path_list


def epoch_name_from_epoch_path(epoch_path: pathlib.Path) -> str:
    return str(epoch_path.stem)


def stacked_fits_path(
    stack_dir_path: pathlib.Path,
    epoch_path: pathlib.Path,
    filter_type: SwiftFilter,
    stacking_method: StackingMethod,
) -> pathlib.Path:
    epoch_name = epoch_name_from_epoch_path(epoch_path=epoch_path)
    filter_string = filter_to_file_string(filter_type)
    fits_filename = f"{epoch_name}_{filter_string}_{stacking_method}.fits"

    return stack_dir_path / pathlib.Path(fits_filename)
