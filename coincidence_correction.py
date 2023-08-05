import math
import numpy as np

from uvot_image import SwiftUVOTImage, SwiftPixelResolution


__all__ = ["coincidence_correction"]


__version__ = "0.0.1"


def coi_factor(rate):
    # correction coefficients
    a1 = -0.0663428
    a2 = 0.0900434
    a3 = -0.0237695
    a4 = -0.0336789
    # dead time, seconds
    df = 0.01577
    # frame time, seconds
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


# here be dragons
def coincidence_correction(img_data: SwiftUVOTImage, scale: SwiftPixelResolution):
    aper = math.ceil(5.0 / float(scale))
    area_frac = np.pi / 4
    # i_ind = np.arange(0,len(img_data))
    # j_ind = np.arange(0,len(img_data[0]))
    # boxlen=aper*2
    # i_ind = (i_ind/boxlen).astype(int)
    # j_ind = (j_ind/boxlen).astype(int)
    # coi_map = pd.DataFrame(img_data,index=i_ind,columns=j_ind)
    # for i in set(coi_map.index):
    #    for j in set(coi_map.columns):
    #        raw = ((coi_map.loc[i,j]).values).sum()*area_frac
    #        factor = coi_factor(raw)
    #        coi_map.loc[i,j]=factor
    # return coi_map.values
    i_len = len(img_data)
    j_len = len(img_data[0])
    vertex = np.zeros((i_len - aper * 2 + 1, j_len - aper * 2 + 1))
    coi_map = np.ones(img_data.shape)
    coi_map_part = np.zeros((i_len - aper * 2, j_len - aper * 2))
    for i in range(0, aper):
        for j in range(0, aper):
            vertex += img_data[
                i : (i_len - 2 * aper + 1 + i), j : (j_len - 2 * aper + 1 + j)
            ]
            vertex += img_data[
                aper + i : (i_len - 2 * aper + 1 + aper + i),
                j : (j_len - 2 * aper + 1 + j),
            ]
            vertex += img_data[
                i : (i_len - 2 * aper + 1 + i),
                aper + j : (j_len - 2 * aper + 1 + aper + j),
            ]
            vertex += img_data[
                aper + i : (i_len - 2 * aper + 1 + aper + i),
                aper + j : (j_len - 2 * aper + 1 + aper + j),
            ]
            # vertex += img_data[(2*aper-1-i):(i_len-i),(2*aper-1-j):(j_len-j)]
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
