from enum import Enum, StrEnum
from dataclasses import dataclass
from typing import TypeAlias, Tuple, Optional

import numpy as np

SwiftUVOTImage: TypeAlias = np.ndarray


# Maps the strings in FITS file header under the keyword DATAMODE
class SwiftImageDataMode(str, Enum):
    data_mode = "IMAGE"
    event_mode = "EVENT"


# The pixel resolution of the given modes according to the swift documentation
class SwiftPixelResolution(float, Enum):
    # units of arcseconds per pixel
    data_mode = 1.0
    event_mode = 0.502


class SwiftUVOTImageType(StrEnum):
    raw = "rw"
    detector = "dt"
    sky_units = "sk"
    exposure_map = "ex"

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]


def datamode_from_fits_keyword_string(datamode: str) -> Optional[SwiftImageDataMode]:
    if datamode == "IMAGE":
        return SwiftImageDataMode.data_mode
    elif datamode == "EVENT":
        return SwiftImageDataMode.event_mode
    else:
        return None


# TODO: clean all of this old code up
# def datamode_to_pixel_resolution(datamode: str) -> SwiftPixelResolution:
#     """Takes a string read from the DATAMODE entry of the FITS header of a UVOT image and converts"""
#     if datamode == "IMAGE":
#         return SwiftPixelResolution.data_mode
#     elif datamode == "EVENT":
#         return SwiftPixelResolution.event_mode
#     else:
#         print(f"Unknown data mode {datamode}! Assuming imaging data mode.")
#         return SwiftPixelResolution.data_mode


def datamode_to_pixel_resolution(datamode: SwiftImageDataMode) -> SwiftPixelResolution:
    if datamode == SwiftImageDataMode.data_mode:
        return SwiftPixelResolution.data_mode
    elif datamode == SwiftImageDataMode.event_mode:
        return SwiftPixelResolution.event_mode


# def pixel_resolution_to_datamode(spr: SwiftPixelResolution) -> str:
#     """Performs the inverse conversion of datamode_to_pixel_resolution for conversion back to a string"""
#     if spr == SwiftPixelResolution.event_mode:
#         return "EVENT"
#     elif spr == SwiftPixelResolution.data_mode:
#         return "IMAGE"


def pixel_resolution_to_datamode(pixel_res: SwiftPixelResolution) -> SwiftImageDataMode:
    if pixel_res == SwiftPixelResolution.data_mode:
        return SwiftImageDataMode.data_mode
    elif pixel_res == SwiftPixelResolution.event_mode:
        return SwiftImageDataMode.event_mode


def float_to_pixel_resolution(pixel_float: float) -> Optional[SwiftPixelResolution]:
    if pixel_float == SwiftPixelResolution.data_mode:
        return SwiftPixelResolution.data_mode
    elif pixel_float == SwiftPixelResolution.event_mode:
        return SwiftPixelResolution.event_mode
    else:
        return None


@dataclass
class PixelCoord:
    """Use floats instead of ints to allow sub-pixel addressing if we need"""

    x: float
    y: float


def get_uvot_image_center_row_col(img: SwiftUVOTImage) -> Tuple[int, int]:
    """Given a SwiftUVOTImage, returns the (row, column) of the center pixel"""
    center_row = int(np.floor(img.shape[0] / 2))
    center_col = int(np.floor(img.shape[1] / 2))
    return (center_row, center_col)


def get_uvot_image_center(img: SwiftUVOTImage) -> PixelCoord:
    x, y = tuple(reversed(get_uvot_image_center_row_col(img=img)))
    return PixelCoord(x=x, y=y)


# def pad_to_match_sizes(
#     uw1: SwiftUVOTImage, uvv: SwiftUVOTImage
# ) -> Tuple[SwiftUVOTImage, SwiftUVOTImage]:
#     """
#     Given two images, pad the smaller image so that the uw1 and uvv images end up the same size
#     """
#     uw1copy = copy.deepcopy(uw1)
#     uvvcopy = copy.deepcopy(uvv)
#
#     cols_to_add = round((uw1.shape[1] - uvv.shape[1]) / 2)
#     rows_to_add = round((uw1.shape[0] - uvv.shape[0]) / 2)
#
#     if cols_to_add > 0:
#         # uw1 is larger, pad uvv to be larger
#         uvv = np.pad(
#             uvv,
#             ((0, 0), (cols_to_add, cols_to_add)),
#             mode="constant",
#             constant_values=0.0,
#         )
#     else:
#         # uvv is larger, pad uw1 to be larger
#         cols_to_add = np.abs(cols_to_add)
#         uw1 = np.pad(
#             uw1,
#             ((0, 0), (cols_to_add, cols_to_add)),
#             mode="constant",
#             constant_values=0.0,
#         )
#
#     if rows_to_add > 0:
#         # uw1 is larger, pad uvv to be larger
#         uvv = np.pad(
#             uvv,
#             ((rows_to_add, rows_to_add), (0, 0)),
#             mode="constant",
#             constant_values=0.0,
#         )
#     else:
#         # uvv is larger, pad uw1 to be larger
#         rows_to_add = np.abs(rows_to_add)
#         uw1 = np.pad(
#             uw1,
#             ((rows_to_add, rows_to_add), (0, 0)),
#             mode="constant",
#             constant_values=0.0,
#         )
#
#     uw1_mid_row_original, uw1_mid_col_original = get_uvot_image_center_row_col(uw1copy)
#     uw1_center_pixel_original = uw1copy[uw1_mid_row_original, uw1_mid_col_original]
#     uw1_mid_row, uw1_mid_col = get_uvot_image_center_row_col(uw1)
#
#     uvv_mid_row_original, uvv_mid_col_original = get_uvot_image_center_row_col(uvvcopy)
#     uvv_center_pixel_original = uvvcopy[uvv_mid_row_original, uvv_mid_col_original]
#     uvv_mid_row, uvv_mid_col = get_uvot_image_center_row_col(uvv)
#
#     pixmatch_list_uw1 = list(zip(*np.where(uw1 == uw1_center_pixel_original)))
#     # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
#     if (uw1_mid_row, uw1_mid_col) not in pixmatch_list_uw1:
#         print("Error padding uw1 image! This is a bug!")
#         print(
#             f"Pixel coordinates of new uw1 image that match center of old uw1 image: {pixmatch_list_uw1}"
#         )
#
#     pixmatch_list_uvv = list(zip(*np.where(uvv == uvv_center_pixel_original)))
#     # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
#     if (uvv_mid_row, uvv_mid_col) not in pixmatch_list_uvv:
#         print("Error padding uvv image! This is a bug!")
#         print(
#             f"Pixel coordinates of new uvv image that match center of old uvv image: {pixmatch_list_uvv}"
#         )
#
#     return (uw1, uvv)
