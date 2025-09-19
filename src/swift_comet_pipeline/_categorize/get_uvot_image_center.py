import numpy as np


from swift_comet_pipeline.scp_types.primitive import *


def get_uvot_image_center_row_col(img: SwiftUvotImage) -> tuple[int, int]:
    """Given a SwiftUvotImage, returns the (row, column) of the center pixel"""
    center_row = int(np.floor(img.shape[0] / 2))
    center_col = int(np.floor(img.shape[1] / 2))
    return (center_row, center_col)


def get_uvot_image_center(img: SwiftUvotImage) -> PixelCoord:
    x, y = tuple(reversed(get_uvot_image_center_row_col(img=img)))
    return PixelCoord(x=x, y=y)
