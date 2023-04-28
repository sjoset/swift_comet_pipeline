import math
import numpy as np

from swift_types import (
    SwiftUVOTImage,
    SwiftPixelResolution,
)


__all__ = ["coincidence_correction"]


__version__ = "0.0.1"


def coi_factor(rate):
    a1 = -0.0663428
    a2 = 0.0900434
    a3 = -0.0237695
    a4 = -0.0336789
    df = 0.01577
    ft = 0.0110329
    raw = rate * (1 - df)
    x = raw * ft
    x2 = x * x
    f = (-np.log(1.0 - x) / (ft * (1.0 - df))) / (
        1.0 + a1 * x + a2 * x2 + a3 * x2 * x + a4 * x2 * x2
    )
    f = f / rate
    f = np.nan_to_num(f, nan=1.0, posinf=1.0, neginf=1.0)
    return f


def coincidence_correction(
    image: SwiftUVOTImage, detector_scale: SwiftPixelResolution
) -> SwiftUVOTImage:
    aper = math.ceil(5.0 / detector_scale.value)
    area_frac = np.pi / 4

    i_len = len(image)
    j_len = len(image[0])
    vertex = np.zeros((i_len - aper * 2 + 1, j_len - aper * 2 + 1))
    coi_map = np.ones(image.shape)
    coi_map_part = np.zeros((i_len - aper * 2, j_len - aper * 2))
    for i in range(0, aper):
        for j in range(0, aper):
            vertex += image[
                i : (i_len - 2 * aper + 1 + i), j : (j_len - 2 * aper + 1 + j)
            ]
            vertex += image[
                aper + i : (i_len - 2 * aper + 1 + aper + i),
                j : (j_len - 2 * aper + 1 + j),
            ]
            vertex += image[
                i : (i_len - 2 * aper + 1 + i),
                aper + j : (j_len - 2 * aper + 1 + aper + j),
            ]
            vertex += image[
                aper + i : (i_len - 2 * aper + 1 + aper + i),
                aper + j : (j_len - 2 * aper + 1 + aper + j),
            ]
            # vertex += image[(2*aper-1-i):(i_len-i),(2*aper-1-j):(j_len-j)]

    vertex = vertex * area_frac
    for i in range(0, 2):
        for j in range(0, 2):
            coi_map_part += vertex[
                i : (i_len - 2 * aper + i), j : (j_len - 2 * aper + j)
            ]
            # coi_map_part += vertex[(1-i):(i_len-2*aper+1-i),(1-j):(j_len-2*aper+1-j)]

    coi_map_part = coi_factor(coi_map_part / 4)
    coi_map[aper : (i_len - aper), aper : (j_len - aper)] = coi_map_part

    return coi_map
