from typing import Tuple

import numpy as np

from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def get_uvot_image_center_row_col(img: SwiftUVOTImage) -> Tuple[int, int]:
    """Given a SwiftUVOTImage, returns the (row, column) of the center pixel"""
    center_row = int(np.floor(img.shape[0] / 2))
    center_col = int(np.floor(img.shape[1] / 2))
    return (center_row, center_col)


def get_uvot_image_center(img: SwiftUVOTImage) -> PixelCoord:
    x, y = tuple(reversed(get_uvot_image_center_row_col(img=img)))
    return PixelCoord(x=x, y=y)
