import numpy as np

from enum import Enum
from dataclasses import dataclass
from typing import TypeAlias, Tuple


__all__ = [
    "SwiftUVOTImage",
    "SwiftUVOTImageType",
    "SwiftPixelResolution",
    "PixelCoord",
    "get_uvot_image_center_row_col",
    "get_uvot_image_center",
]


SwiftUVOTImage: TypeAlias = np.ndarray


class SwiftPixelResolution(str, Enum):
    event_mode = 0.502
    data_mode = 1.0


class SwiftUVOTImageType(str, Enum):
    raw = "rw"
    detector = "dt"
    sky_units = "sk"
    exposure_map = "ex"

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]


def datamode_to_pixel_resolution(datamode: str) -> SwiftPixelResolution:
    if datamode == "IMAGE":
        return SwiftPixelResolution.data_mode
    elif datamode == "EVENT":
        return SwiftPixelResolution.event_mode
    else:
        print(f"Unknown data mode {datamode}!")
        return SwiftPixelResolution.data_mode


def pixel_resolution_to_datamode(spr: SwiftPixelResolution) -> str:
    if spr == SwiftPixelResolution.event_mode:
        return "EVENT"
    elif spr == SwiftPixelResolution.data_mode:
        return "IMAGE"


@dataclass
class PixelCoord:
    """Use floats instead of ints to allow sub-pixel addressing if we need"""

    x: float
    y: float


def get_uvot_image_center_row_col(img: SwiftUVOTImage) -> Tuple[int, int]:
    center_row = int(np.floor(img.shape[0] / 2))
    center_col = int(np.floor(img.shape[1] / 2))
    return (center_row, center_col)


def get_uvot_image_center(img: SwiftUVOTImage) -> PixelCoord:
    x, y = tuple(reversed(get_uvot_image_center_row_col(img=img)))
    return PixelCoord(x=x, y=y)
