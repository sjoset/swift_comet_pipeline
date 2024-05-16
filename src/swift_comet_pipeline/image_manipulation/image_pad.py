import copy
from typing import Tuple

import numpy as np

from swift_comet_pipeline.swift.uvot_image import (
    SwiftUVOTImage,
    get_uvot_image_center_row_col,
)


def pad_to_match_sizes(
    img_one: SwiftUVOTImage, img_two: SwiftUVOTImage
) -> Tuple[SwiftUVOTImage, SwiftUVOTImage]:
    """
    Given two images, zero-pad the edges of the smaller image so that both images end up the same width and
    have an odd number of columns.
    Does the same for the height, guaranteeing on odd number of rows in the resulting images.
    Preserves the center of each image, so that the new images only have new (zero) pixels at their edges.
    """
    img_one_copy = copy.deepcopy(img_one)
    img_two_copy = copy.deepcopy(img_two)

    cols_to_add = round((img_one.shape[1] - img_two.shape[1]) / 2)
    rows_to_add = round((img_one.shape[0] - img_two.shape[0]) / 2)

    if cols_to_add > 0:
        # img_one is larger, pad img_two to be larger
        img_two = np.pad(
            img_two,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        # img_two is larger, pad img_one to be larger
        cols_to_add = np.abs(cols_to_add)
        img_one = np.pad(
            img_one,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )

    if rows_to_add > 0:
        # img_one is larger, pad img_two to be larger
        img_two = np.pad(
            img_two,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        # img_two is larger, pad img_one to be larger
        rows_to_add = np.abs(rows_to_add)
        img_one = np.pad(
            img_one,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    img_one_mid_row_original, img_one_mid_col_original = get_uvot_image_center_row_col(
        img_one_copy
    )
    img_one_center_pixel_original = img_one_copy[
        img_one_mid_row_original, img_one_mid_col_original
    ]
    img_one_mid_row, img_one_mid_col = get_uvot_image_center_row_col(img_one)

    img_two_mid_row_original, img_two_mid_col_original = get_uvot_image_center_row_col(
        img_two_copy
    )
    img_two_center_pixel_original = img_two_copy[
        img_two_mid_row_original, img_two_mid_col_original
    ]
    img_two_mid_row, img_two_mid_col = get_uvot_image_center_row_col(img_two)

    pixmatch_list_img_one = list(
        zip(*np.where(img_one == img_one_center_pixel_original))
    )
    # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
    if (img_one_mid_row, img_one_mid_col) not in pixmatch_list_img_one:
        print("Error padding uw1 image! This is a bug!")
        print(
            f"Pixel coordinates of new uw1 image that match center of old uw1 image: {pixmatch_list_img_one}"
        )

    pixmatch_list_img_two = list(
        zip(*np.where(img_two == img_two_center_pixel_original))
    )
    # the center pixel of the new image should match the center pixel of the original - so it should be in this list!
    if (img_two_mid_row, img_two_mid_col) not in pixmatch_list_img_two:
        print("Error padding uvv image! This is a bug!")
        print(
            f"Pixel coordinates of new uvv image that match center of old uvv image: {pixmatch_list_img_two}"
        )

    return (img_one, img_two)
