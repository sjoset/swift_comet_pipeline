import pandas as pd

from swift_comet_pipeline.swift.uvot_image import PixelCoord


def invalid_user_center_value() -> float:
    """
    Pixel coordinate value stored in the observation log when the user has not manually adjusted the comet center for an image
    """
    return -1


def has_valid_user_center_value(row: pd.Series) -> bool:
    """
    Takes an observation log dataframe row or Epoch dataframe row and returns if
    the user has specified a comet center or not in the veto pipeline step
    """

    if (
        row.USER_CENTER_X != invalid_user_center_value()
        and row.USER_CENTER_Y != invalid_user_center_value()
    ):
        return True
    else:
        return False


def get_horizons_comet_center(row: pd.Series) -> PixelCoord:
    """
    Takes an observation log dataframe row or Epoch dataframe row and returns a PixelCoord of the comet center
    as specified by JPL Horizons
    """
    return PixelCoord(x=row.PX, y=row.PY)


def get_user_specified_comet_center(row: pd.Series) -> PixelCoord | None:
    """
    Takes an observation log dataframe row or Epoch dataframe row and returns either a PixelCoord of the comet
    center the user specified in the veto step, or returns None
    """
    if has_valid_user_center_value(row=row):
        return PixelCoord(x=row.USER_CENTER_X, y=row.USER_CENTER_Y)
    else:
        return None


def get_comet_center_prefer_user_coords(row: pd.Series) -> PixelCoord:
    """
    Takes an observation log dataframe row or Epoch dataframe row and returns a PixelCoord of the comet center

    If a user-specified comet center exists, we prefer that over the JPL Horizons value
    If no user-specified comet center exists, return the JPL Horizons value
    """
    comet_center_coords = get_user_specified_comet_center(row=row)
    if comet_center_coords is None:
        comet_center_coords = get_horizons_comet_center(row=row)
    return comet_center_coords
