import numpy as np

from photutils.aperture import CircularAperture, ApertureStats

from enum import StrEnum, auto
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string

from swift_comet_pipeline.swift.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)


class CometCenterFindingMethod(StrEnum):
    pixel_center = auto()
    aperture_centroid = auto()
    aperture_peak = auto()

    @classmethod
    def all_methods(cls):
        return [x for x in cls]


def find_comet_center(
    img: SwiftUVOTImage,
    method: CometCenterFindingMethod = CometCenterFindingMethod.pixel_center,
    search_aperture: CircularAperture | None = None,
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
    img: SwiftUVOTImage, search_aperture: CircularAperture | None
) -> PixelCoord:
    if search_aperture is None:
        print("No aperture provided for center finding by centroid!")
        return PixelCoord(-1.0, -1.0)

    stats = ApertureStats(img, search_aperture)

    return PixelCoord(x=stats.centroid[0], y=stats.centroid[1])


def comet_center_by_peak(
    img: SwiftUVOTImage, search_aperture: CircularAperture | None
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


# TODO: deprecate?
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
