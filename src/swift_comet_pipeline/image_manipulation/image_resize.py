from swift_comet_pipeline.scp_types.primitive import *


def stretch_image(img: SwiftUvotImage, stretch_factor: int) -> SwiftUvotImage:

    stretched_img = img.repeat(stretch_factor, axis=0).repeat(stretch_factor, axis=1)

    return stretched_img
