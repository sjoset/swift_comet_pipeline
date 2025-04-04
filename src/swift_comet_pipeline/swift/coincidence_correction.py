import copy

import numpy as np
from scipy.signal import convolve2d

from swift_comet_pipeline.types.coincidence_correction import CoincidenceCorrection
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def coincidence_correction(
    img: SwiftUVOTImage, scale: SwiftPixelResolution
) -> np.ndarray:

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


# def coincidence_correction_old(img: SwiftUVOTImage, scale: SwiftPixelResolution):
#     img_data = copy.deepcopy(img)
#
#     # the padding around the images from swift are pure zeros - we have to divide by the pixel value
#     # in CoincidenceCorrection.coi_factor() so change the padding pixels to be slightly non-zero
#     dead_space_mask = img_data == 0
#     img_data[dead_space_mask] = 1e-29
#
#     coi = CoincidenceCorrection()
#     aper = math.ceil(5.0 / float(scale))
#     area_frac = np.pi / 4
#     i_len = len(img_data)
#     j_len = len(img_data[0])
#     vertex = np.zeros((i_len - aper * 2 + 1, j_len - aper * 2 + 1))
#     coi_map = np.ones(img_data.shape)
#     coi_map_part = np.zeros((i_len - aper * 2, j_len - aper * 2))
#     for i in range(0, aper):
#         for j in range(0, aper):
#             vertex += img_data[
#                 i : (i_len - 2 * aper + 1 + i), j : (j_len - 2 * aper + 1 + j)
#             ]
#             vertex += img_data[
#                 aper + i : (i_len - 2 * aper + 1 + aper + i),
#                 j : (j_len - 2 * aper + 1 + j),
#             ]
#             vertex += img_data[
#                 i : (i_len - 2 * aper + 1 + i),
#                 aper + j : (j_len - 2 * aper + 1 + aper + j),
#             ]
#             vertex += img_data[
#                 aper + i : (i_len - 2 * aper + 1 + aper + i),
#                 aper + j : (j_len - 2 * aper + 1 + aper + j),
#             ]
#             # vertex += img_data[(2*aper-1-i):(i_len-i),(2*aper-1-j):(j_len-j)]
#     vertex = vertex * area_frac
#     for i in range(0, 2):
#         for j in range(0, 2):
#             coi_map_part += vertex[
#                 i : (i_len - 2 * aper + i), j : (j_len - 2 * aper + j)
#             ]
#             # coi_map_part += vertex[(1-i):(i_len-2*aper+1-i),(1-j):(j_len-2*aper+1-j)]
#     coi_map_part = coi.coi_factor(coi_map_part / 4)  # type: ignore
#     coi_map[aper : (i_len - aper), aper : (j_len - aper)] = coi_map_part
#     return coi_map
