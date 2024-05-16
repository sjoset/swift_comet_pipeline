from typing import Tuple

import numpy as np
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt

from swift_comet_pipeline.swift.uvot_image import PixelCoord, SwiftUVOTImage


def get_image_dimensions_to_center_on_pixel(
    source_image: SwiftUVOTImage, coords_to_center: PixelCoord
) -> Tuple[float, float]:
    """
    If we want to re-center the source_image on coords_to_center, we need to figure out the dimensions the new image
    needs to be to fit the old picture after we shift it over to its new position
    Returns tuple of (rows, columns) the new image would have to be
    """
    num_rows, num_columns = source_image.shape
    center_row = num_rows / 2.0
    center_col = num_columns / 2.0

    # distance the image needs to move
    d_rows = center_row - coords_to_center.y
    d_cols = center_col - coords_to_center.x

    # the total padding is twice the distance it needs to move -
    # extra space for it to move into, and the empty space it would leave behind
    row_padding = 2 * d_rows
    col_padding = 2 * d_cols

    new_rows_cols = (
        np.ceil(source_image.shape[0] + np.abs(row_padding)),
        np.ceil(source_image.shape[1] + np.abs(col_padding)),
    )

    return new_rows_cols


def center_image_on_coords(
    source_image: SwiftUVOTImage,
    source_coords_to_center: PixelCoord,
    stacking_image_size: Tuple[int, int],
    show_resulting_image: bool = False,
) -> SwiftUVOTImage:
    """
    Takes a source_image and pads with zeros so that the new image has source_coords_to_center
    """

    center_x, center_y = np.round(source_coords_to_center.x), np.round(
        source_coords_to_center.y
    )
    new_r, new_c = map(lambda x: int(x), np.ceil(stacking_image_size))

    # enforce that we have an odd number of pixels so that the comet can be moved to the center row, center column
    if new_r % 2 == 0:
        half_r = new_r / 2
        new_r = new_r + 1
    else:
        half_r = (new_r - 1) / 2
    if new_c % 2 == 0:
        half_c = new_c / 2
        new_c = new_c + 1
    else:
        half_c = (new_c - 1) / 2

    # create empty array to hold the new, centered image
    centered_image = np.zeros((new_r, new_c))

    def shift_row(r):
        return int(r + half_r + 0 - center_y)

    def shift_column(c):
        return int(c + half_c + 0 - center_x)

    for r in range(source_image.shape[0]):
        for c in range(source_image.shape[1]):
            centered_image[shift_row(r), shift_column(c)] = source_image[r, c]

    # log.debug(">> center_image_on_coords: image center at (%s, %s)", half_c, half_r)
    # log.debug(
    #     ">> center_image_on_coords: Shifted comet center to coordinates (%s, %s): this should match image center",
    #     shift_column(center_x),
    #     shift_row(center_y),
    # )

    if show_resulting_image:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(centered_image)
        ax.imshow(centered_image, vmin=vmin, vmax=vmax, origin="lower")
        ax.axvline(int(np.floor(centered_image.shape[1] / 2)), color="b", alpha=0.1)
        ax.axhline(int(np.floor(centered_image.shape[0] / 2)), color="b", alpha=0.1)
        plt.show()

    return centered_image
