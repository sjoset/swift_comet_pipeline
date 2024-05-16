from typing import Optional

import pandas as pd

from swift_comet_pipeline.swift.uvot_image import PixelCoord


def invalid_user_center_value() -> float:
    """
    Pixel value stored in the observation log when the user has not manually adjusted the comet center for an image
    """
    return -1


def has_valid_user_center_value(row: pd.Series) -> bool:

    if (
        row.USER_CENTER_X != invalid_user_center_value()
        and row.USER_CENTER_Y != invalid_user_center_value()
    ):
        return True
    else:
        return False


def get_horizons_comet_center(row: pd.Series) -> PixelCoord:
    return PixelCoord(x=row.PX, y=row.PY)


def get_user_specified_comet_center(row: pd.Series) -> Optional[PixelCoord]:
    if has_valid_user_center_value(row=row):
        return PixelCoord(x=row.USER_CENTER_X, y=row.USER_CENTER_Y)
    else:
        return None


def get_comet_center_prefer_user_coords(row: pd.Series) -> PixelCoord:
    comet_center_coords = get_user_specified_comet_center(row=row)
    if comet_center_coords is None:
        comet_center_coords = get_horizons_comet_center(row=row)
    return comet_center_coords
