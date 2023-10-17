import numpy as np

from photutils.aperture import CircularAperture, ApertureStats

from typing import Optional
from enum import StrEnum, auto

from uvot_image import PixelCoord, SwiftUVOTImage, get_uvot_image_center
from count_rate import CountRate, CountRatePerPixel


__all__ = [
    "CometCenterFindingMethod",
    "comet_manual_aperture",
    "find_comet_center",
]


class CometCenterFindingMethod(StrEnum):
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
    bg: CountRatePerPixel,
) -> CountRate:
    comet_aperture = CircularAperture((aperture_x, aperture_y), r=aperture_radius)
    comet_aperture_stats = ApertureStats(img, comet_aperture)

    comet_count_rate = float(comet_aperture_stats.sum)
    # TODO: this is not a good error calculation but it will have to do for now
    comet_count_rate_sigma = comet_aperture_stats.std

    propogated_sigma = np.sqrt(
        comet_count_rate_sigma**2
        + (comet_aperture_stats.sum_aper_area.value * bg.sigma**2)
    )

    return CountRate(value=comet_count_rate, sigma=propogated_sigma)


def find_comet_center(
    img: SwiftUVOTImage,
    method: CometCenterFindingMethod,
    search_aperture: Optional[CircularAperture] = None,
) -> PixelCoord:
    """
    Coordinates returned are x, y values
    """
    if method == CometCenterFindingMethod.pixel_center:
        return get_uvot_image_center(img=img)
    elif method == CometCenterFindingMethod.aperture_centroid:
        return comet_center_by_centroid(img=img, search_aperture=search_aperture)
    elif method == CometCenterFindingMethod.aperture_peak:
        return comet_center_by_peak(img=img, search_aperture=search_aperture)


def comet_center_by_centroid(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> PixelCoord:
    if search_aperture is None:
        print("No aperture provided for center finding by centroid!")
        return PixelCoord(-1.0, -1.0)

    stats = ApertureStats(img, search_aperture)

    # return tuple(stats.centroid)
    return PixelCoord(x=stats.centroid[0], y=stats.centroid[1])


def comet_center_by_peak(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> PixelCoord:
    if search_aperture is None:
        print("No aperture provided for center finding by peak!")
        return PixelCoord(-1.0, -1.0)

    # cut out the pixels in the aperture
    ap_mask = search_aperture.to_mask(method="center")
    img_cutout = ap_mask.cutout(data=img)  # type: ignore

    # index of peak value for img represented as 1d list
    peak_pos_raveled = np.argmax(img_cutout)
    # unravel turns this 1d index into (row, col) indices
    peak_pos = np.unravel_index(peak_pos_raveled, img_cutout.shape)

    ap_min_x, ap_min_y = search_aperture.bbox.ixmin, search_aperture.bbox.iymin  # type: ignore

    # return (float(ap_min_x + peak_pos[1]), float(ap_min_y + peak_pos[0]))
    return PixelCoord(x=ap_min_x + peak_pos[1], y=ap_min_y + peak_pos[0])
