from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def trim_image_and_relocate_pixel_coords(
    img: SwiftUVOTImage, x_min: int, x_max: int, y_min: int, y_max: int, pc: PixelCoord
) -> tuple[SwiftUVOTImage, PixelCoord]:
    """
    Trim down the given image, and move the coordinate pc based on the trim to point to the same pixel
    """

    new_img = img[y_min:y_max, x_min:x_max].copy()
    new_pixel_coord = PixelCoord(x=pc.x - x_min, y=pc.y - y_min)

    return new_img, new_pixel_coord
