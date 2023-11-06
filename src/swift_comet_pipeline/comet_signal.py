import numpy as np

from photutils.aperture import CircularAperture, ApertureStats

from typing import Optional
from enum import StrEnum, auto
from swift_comet_pipeline.swift_filter import SwiftFilter, filter_to_file_string

from swift_comet_pipeline.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)
from swift_comet_pipeline.count_rate import CountRate, CountRatePerPixel


__all__ = [
    "CometCenterFindingMethod",
    "comet_manual_aperture",
    "find_comet_center",
    "compare_comet_center_methods",
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


def compare_comet_center_methods(uw1: SwiftUVOTImage, uvv: SwiftUVOTImage):
    # TODO: uvv images tend to pick up the dust tail so we might expect some difference depending on the method of center detection,
    # so maybe we just use the uw1 and assume it's less likely to have a tail to scramble the center-finding

    peaks = {}
    imgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        print(f"Determining center of comet for {filter_to_file_string(filter_type)}:")
        pix_center = get_uvot_image_center(img=imgs[filter_type])
        search_ap = CircularAperture((pix_center.x, pix_center.y), r=30)
        pixel_center = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.pixel_center,
            search_aperture=search_ap,
        )
        centroid = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.aperture_centroid,
            search_aperture=search_ap,
        )
        peak = find_comet_center(
            img=imgs[filter_type],
            method=CometCenterFindingMethod.aperture_peak,
            search_aperture=search_ap,
        )
        print("\tBy image center: ", pixel_center)
        print(
            "\tBy centroid (center of mass) in aperture radius 30 at image center: ",
            centroid,
        )
        print("\tBy peak value in aperture radius 30 at image center: ", peak)

        peaks[filter_type] = peak

    xdist = peaks[SwiftFilter.uw1].x - peaks[SwiftFilter.uvv].x
    ydist = peaks[SwiftFilter.uw1].y - peaks[SwiftFilter.uvv].y
    dist = np.sqrt(xdist**2 + ydist**2)
    if dist > np.sqrt(2.0):
        print(
            f"Comet peaks in uw1 and uvv are separated by {dist} pixels! Fitting might suffer."
        )

    # show_centers(uw1, [pixel_center, centroid, peak])  # pyright: ignore


# def show_centers(img, cs: List[PixelCoord]):
#     # img_scaled = np.log10(img)
#     img_scaled = img
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#
#     pix_center = get_uvot_image_center(img)
#     ax1.add_patch(
#         plt.Circle(
#             (pix_center.x, pix_center.y),
#             radius=30,
#             fill=False,
#         )
#     )
#
#     zscale = ZScaleInterval()
#     vmin, vmax = zscale.get_limits(img_scaled)
#
#     im1 = ax1.imshow(img_scaled, vmin=vmin, vmax=vmax)
#     fig.colorbar(im1)
#
#     for c in cs:
#         line_color = next(ax1._get_lines.prop_cycler)["color"]
#         ax1.axvline(c.x, alpha=0.7, color=line_color)
#         ax1.axhline(c.y, alpha=0.9, color=line_color)
#
#     plt.show()
