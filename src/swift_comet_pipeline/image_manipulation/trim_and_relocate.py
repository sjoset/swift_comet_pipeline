from swift_comet_pipeline.scp_types.primitive import *


def trim_image_and_relocate_pixel_coords(
    img: SwiftUvotImage, x_min: int, x_max: int, y_min: int, y_max: int, pc: PixelCoord
) -> tuple[SwiftUvotImage, PixelCoord]:
    """
    Trim down the given image, and return a new PixelCoord pc' that points to the same
    pixel as pc, but after the trim
    """

    new_img = img[y_min:y_max, x_min:x_max].copy()
    new_pixel_coord = PixelCoord(x=pc.x - x_min, y=pc.y - y_min)

    return new_img, new_pixel_coord


def trim_image(
    img: SwiftUvotImage, left_right_pix: int, top_bottom_pix: int
) -> SwiftUvotImage:
    # trim an equal amount of pixels from left/right, top/bottom, or both
    h, w = img.shape
    return img[top_bottom_pix : h - top_bottom_pix, left_right_pix : w - left_right_pix]
