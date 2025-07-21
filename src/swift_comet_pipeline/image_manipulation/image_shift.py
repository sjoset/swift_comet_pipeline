import scipy

from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def shift_image_keep_dimensions(
    img: SwiftUVOTImage, shift_x: int, shift_y: int, fill_value=0
) -> SwiftUVOTImage:
    """
    Moves the pixels of the image by shift_x pixels and shift_y pixels, while keeping the resulting image the same size.
    Newly created pixels are filled with fill_value.
    """

    shifted_img = scipy.ndimage.shift(
        img,
        shift=(shift_y, shift_x),
        order=0,
        mode="constant",
        cval=fill_value,
        prefilter=False,
    )
    return shifted_img
