from swift_comet_pipeline.image_manipulation.image_recenter import (
    get_image_dimensions_to_center_on_pixel,
)
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.stacking import StackableUVOTImage
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def determine_stacking_image_size(
    img_list: list[SwiftUVOTImage], comet_center_coords: list[PixelCoord]
) -> tuple[int, int] | None:
    """
    Examines every image and finds the image size necessary to accommodate
    the largest image involved in the stack, so we can pad out the smaller images and stack them in one step
    """

    # stores how big each image would need to be if recentered on the comet
    recentered_image_dimensions_rows_cols = []

    recentered_image_dimensions_rows_cols = [
        get_image_dimensions_to_center_on_pixel(source_image=img, coords_to_center=cc)
        for img, cc in zip(img_list, comet_center_coords)
    ]

    if len(recentered_image_dimensions_rows_cols) == 0:
        print("No images found in epoch!")
        return None

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(
        recentered_image_dimensions_rows_cols, key=lambda k: k[0], reverse=True
    )[0][0]
    max_num_cols = sorted(
        recentered_image_dimensions_rows_cols, key=lambda k: k[1], reverse=True
    )[0][1]

    return (int(max_num_rows), int(max_num_cols))


def determine_stacking_image_size_from_stackables(
    stackables: list[StackableUVOTImage],
) -> tuple[int, int] | None:
    """
    Examines every image and finds the image size necessary to accommodate
    the largest image involved in the stack, so we can pad out the smaller images and stack them in one step
    """

    img_list = [s.img for s in stackables]
    comet_center_coords = [s.comet_center for s in stackables]
    return determine_stacking_image_size(
        img_list=img_list, comet_center_coords=comet_center_coords
    )

    # # stores how big each image would need to be if recentered on the comet
    # recentered_image_dimensions_rows_cols = []
    #
    # recentered_image_dimensions_rows_cols = [
    #     get_image_dimensions_to_center_on_pixel(
    #         source_image=s.img, coords_to_center=s.comet_center
    #     )
    #     for s in stackables
    # ]
    #
    # if len(recentered_image_dimensions_rows_cols) == 0:
    #     print("No images found in epoch!")
    #     return None
    #
    # # now take the largest size so that every image can be stacked without losing pixels
    # max_num_rows = sorted(
    #     recentered_image_dimensions_rows_cols, key=lambda k: k[0], reverse=True
    # )[0][0]
    # max_num_cols = sorted(
    #     recentered_image_dimensions_rows_cols, key=lambda k: k[1], reverse=True
    # )[0][1]
    #
    # return (int(max_num_rows), int(max_num_cols))
