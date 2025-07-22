import numpy as np
from photutils.aperture import ApertureStats, CircularAperture

from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


# TODO:
# We need three functions: total, mean, and median count rate


def aperture_total_count_rate(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
    background: BackgroundResult | None,
    exposure_time_s: float,
) -> CountRate:
    """
    Constructs a circular aperture of aperture_radius at aperture_center and takes the sum of pixels
    inside as the count rate, along with its error
    If a background has not been subtracted, set background to None
    Otherwise, assumes the background has been subtracted
    """

    ap = CircularAperture((aperture_center.x, aperture_center.y), r=aperture_radius)
    ap_stats = ApertureStats(img, ap)

    total_ap_count_rate = float(ap_stats.sum)
    aperture_area_pix = ap_stats.sum_aper_area.value

    source_variance = total_ap_count_rate / exposure_time_s

    if background is not None:
        k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
        bg_variance_per_pixel = background.count_rate_per_pixel.sigma**2
        bg_area = background.bg_aperture_area

        bg_variance = (
            aperture_area_pix
            * bg_variance_per_pixel
            * (1 + (k * aperture_area_pix) / (exposure_time_s * bg_area))
        )
    else:
        bg_variance = 0.0

    total_variance = source_variance + bg_variance

    return CountRate(total_ap_count_rate, sigma=np.sqrt(total_variance))


# def aperture_count_rate(
#     img: SwiftUVOTImage,
#     aperture_center: PixelCoord,
#     aperture_radius: float,
#     bg: CountRatePerPixel,
# ) -> CountRate:
#     """
#     Constructs a circular aperture of aperture_radius at aperture_center and takes the sum of pixels
#     inside as the count rate, along with its error
#     """
#     comet_aperture = CircularAperture(
#         (aperture_center.x, aperture_center.y), r=aperture_radius
#     )
#     comet_aperture_stats = ApertureStats(img, comet_aperture)
#
#     comet_count_rate = float(comet_aperture_stats.sum)
#     # TODO: this is not a good error calculation but it will have to do for now
#     # TODO: use calc_total_error here
#     comet_count_rate_sigma = comet_aperture_stats.std
#
#     propogated_sigma = np.sqrt(
#         comet_count_rate_sigma**2
#         + (comet_aperture_stats.sum_aper_area.value * bg.sigma**2)
#     )
#
#     return CountRate(value=comet_count_rate, sigma=propogated_sigma)
