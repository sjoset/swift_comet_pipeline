from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage


def stretch_image(img: SwiftUVOTImage, stretch_factor: int) -> SwiftUVOTImage:

    stretched_img = img.repeat(stretch_factor, axis=0).repeat(stretch_factor, axis=1)

    return stretched_img
