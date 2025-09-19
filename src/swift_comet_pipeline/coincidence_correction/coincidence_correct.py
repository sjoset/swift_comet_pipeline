import copy

import numpy as np
from scipy.signal import convolve2d

from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.compound.coincidence_correction import (
    CoincidenceCorrection,
)


def coincidence_correction_factor_map(
    img: SwiftUvotImage, scale: UvotPixelResolution
) -> np.ndarray:
    """
    Returns an array the same size as img, intended to be multiplied by the image's pixels
    to perform coincidence correction
    """

    # make a copy so we can apply coincidence correction without altering original
    img_data = copy.deepcopy(img)
    # Replace the padding pixels (zeros) with a very small value to avoid division by zero
    dead_space_mask = img_data == 0
    img_data[dead_space_mask] = 1e-29

    coi = CoincidenceCorrection()
    kernel = coi.kernel[scale]

    with np.errstate(divide="ignore", invalid="ignore"):
        coi_map: SwiftUVOTImage = coi.coi_factor(convolve2d(img_data, kernel, mode="same"))  # type: ignore

    # Replace the zero pixels with ones: we multiply the image by coi_map, so this means no correction
    zeros_mask = coi_map == 0.0
    coi_map[zeros_mask] = 1.0
    return coi_map
