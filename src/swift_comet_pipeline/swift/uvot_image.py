from enum import Enum, StrEnum
from dataclasses import dataclass
from typing import TypeAlias, Tuple

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


def datamode_from_fits_keyword_string(datamode: str) -> SwiftImageDataMode | None:
    if datamode == "IMAGE":
        return SwiftImageDataMode.data_mode
    elif datamode == "EVENT":
        return SwiftImageDataMode.event_mode
    else:
        return None


def datamode_to_pixel_resolution(datamode: SwiftImageDataMode) -> SwiftPixelResolution:
    if datamode == SwiftImageDataMode.data_mode:
        return SwiftPixelResolution.data_mode
    elif datamode == SwiftImageDataMode.event_mode:
        return SwiftPixelResolution.event_mode


def pixel_resolution_to_datamode(pixel_res: SwiftPixelResolution) -> SwiftImageDataMode:
    if pixel_res == SwiftPixelResolution.data_mode:
        return SwiftImageDataMode.data_mode
    elif pixel_res == SwiftPixelResolution.event_mode:
        return SwiftImageDataMode.event_mode


def float_to_pixel_resolution(pixel_float: float) -> SwiftPixelResolution | None:
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
