import numpy as np
from photutils.aperture import CircularAperture, ApertureStats

from typing import Tuple, Optional
from enum import Enum, auto

from uvot_image import SwiftUVOTImage, get_uvot_image_center


__all__ = ["CometCenterFindingMethod", "comet_manual_aperture", "find_comet_center"]


class CometCenterFindingMethod(str, Enum):
    pixel_center = auto()
    aperture_centroid = auto()
    aperture_peak = auto()

    @classmethod
    def all_methods(cls):
        return [x for x in cls]


def comet_manual_aperture(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> float:
    comet_aperture = CircularAperture((aperture_x, aperture_y), r=aperture_radius)
    comet_aperture_stats = ApertureStats(img, comet_aperture)
    comet_count_rate = float(comet_aperture_stats.sum)

    return comet_count_rate


# TODO: this should return PixelCoord
def find_comet_center(
    img: SwiftUVOTImage,
    method: CometCenterFindingMethod,
    search_aperture: Optional[CircularAperture] = None,
) -> Tuple[float, float]:
    """
    Coordinates returned are x, y values
    """
    if method == CometCenterFindingMethod.pixel_center:
        # center_row_int, center_col_int = get_uvot_image_center_row_col(img)
        pix_center = get_uvot_image_center(img=img)
        return (pix_center.x, pix_center.y)
    elif method == CometCenterFindingMethod.aperture_centroid:
        return comet_center_by_centroid(img=img, search_aperture=search_aperture)
    elif method == CometCenterFindingMethod.aperture_peak:
        return comet_center_by_peak(img=img, search_aperture=search_aperture)


def comet_center_by_centroid(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> Tuple[float, float]:
    if search_aperture is None:
        print("No aperture provided for center finding by centroid!")
        return (0.0, 0.0)

    stats = ApertureStats(img, search_aperture)

    return tuple(stats.centroid)


def comet_center_by_peak(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> Tuple[float, float]:
    if search_aperture is None:
        print("No aperture provided for center finding by peak!")
        return (0.0, 0.0)

    # cut out the pixels in the aperture
    ap_mask = search_aperture.to_mask(method="center")
    img_cutout = ap_mask.cutout(data=img)  # type: ignore

    # index of peak value for img represented as 1d list
    peak_pos_raveled = np.argmax(img_cutout)
    # unravel turns this 1d index into (row, col) indices
    peak_pos = np.unravel_index(peak_pos_raveled, img_cutout.shape)

    ap_min_x, ap_min_y = search_aperture.bbox.ixmin, search_aperture.bbox.iymin  # type: ignore

    return (float(ap_min_x + peak_pos[1]), float(ap_min_y + peak_pos[0]))
