import numpy as np
from photutils.aperture import Aperture, ApertureStats

from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


# TODO:
# We need three functions: total, mean, and median count rate


# def total_count_rate_at_circular_aperture(
#     img: SwiftUVOTImage,
#     aperture_center: PixelCoord,
#     aperture_radius: float,
#     background: BackgroundResult | None,
#     exposure_time_s: float,
# ) -> CountRate:
#
#     ap = CircularAperture((aperture_center.x, aperture_center.y), r=aperture_radius)
#     return total_aperture_count_rate(
#         img=img, ap=ap, background=background, exposure_time_s=exposure_time_s
#     )


def total_aperture_count_rate(
    img: SwiftUVOTImage,
    ap: Aperture,
    background: BackgroundResult | None,
    exposure_time_s: float,
) -> CountRate:
    """
    Takes the sum of pixels inside as the count rate, along with its error
    If a background has not been subtracted, set background to None
    Otherwise, assumes the background has been subtracted from 'img'
    """

    ap_stats = ApertureStats(img, ap)

    total_ap_count_rate = float(ap_stats.sum)
    aperture_area_pix = ap_stats.sum_aper_area.value

    source_variance = total_ap_count_rate / exposure_time_s

    # net negative count rate - estimate the error
    if source_variance < 0.0:
        source_variance = np.abs(source_variance)

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


def median_aperture_count_rate(
    img: SwiftUVOTImage,
    ap: Aperture,
    background: BackgroundResult | None,
    exposure_time_s: float,
) -> CountRate:
    """
    Constructs a circular aperture of aperture_radius at aperture_center and takes the sum of pixels
    inside as the count rate, along with its error
    If a background has not been subtracted, set background to None
    Otherwise, assumes the background has been subtracted
    """

    ap_stats = ApertureStats(img, ap)

    aperture_area_pix = ap_stats.sum_aper_area.value

    # estimate variance from MAD - median absolute deviation
    mad = float(ap_stats.mad_std)

    source_variance = (np.pi / 2) * mad**2 / (aperture_area_pix * exposure_time_s)

    # net negative count rate - estimate the error
    if source_variance < 0.0:
        source_variance = np.abs(source_variance)

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

    return CountRate(float(ap_stats.median), sigma=np.sqrt(total_variance))
