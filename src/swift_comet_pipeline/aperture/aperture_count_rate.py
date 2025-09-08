from dataclasses import dataclass

import numpy as np
from photutils.aperture import Aperture, ApertureStats
from photutils.aperture.stats import SigmaClip

from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


# TODO: move this to types
@dataclass
class ApertureCountRateAnalysis:
    """
    Given an aperture, we want these results for determining the signal within it
    """

    total_count_rate: float
    median_count_rate: float
    mean_count_rate: float
    # calculated with the sum of the count rates over the exposure time
    count_rate_shot_noise_variance: float
    # what variance the background contributed, total
    bg_variance: float
    # total variance of the sum: count_rate_shot_noise_variance + bg_variance
    total_count_rate_variance: float
    # variance if we use the median

    # if any sigma clipping was used
    sigma_clip: float
    # total valid pixels left in aperture for signal - sigma clipping may remove some
    ap_num_pixels: float


def aperture_analysis(
    img: SwiftUVOTImage,
    ap: Aperture,
    background: BackgroundResult | None,
    exposure_time_s: float,
    sigma_clip: float = 3.0,
) -> ApertureCountRateAnalysis:
    """
    Takes an aperture and returns ApertureCountRateAnalysis
    If a background has not been subtracted, set background to None
    Otherwise, assumes the background has been subtracted, and uses 'background' to factor in bg error
    """

    ap_stats = ApertureStats(
        img, ap, sigma_clip=SigmaClip(sigma=sigma_clip, cenfunc="median")
    )

    # count rate measures
    total_count_rate: float = ap_stats.sum  # type: ignore
    median_count_rate: float = ap_stats.median  # type: ignore
    mean_count_rate: float = ap_stats.mean  # type: ignore
    ap_num_pixels: float = ap_stats.sum_aper_area.value

    # variances
    count_rate_shot_noise_variance: float = ap_stats.sum / exposure_time_s  # type: ignore
    # net negative count rate - estimate the error
    if count_rate_shot_noise_variance < 0.0:
        count_rate_shot_noise_variance = np.abs(count_rate_shot_noise_variance)

    if background is None:
        bg_variance = 0.0
    else:
        k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
        bg_shot_noise_variance = background.bg_shot_noise_variance
        bg_area = background.bg_num_pixels

        bg_variance = (
            ap_num_pixels * bg_shot_noise_variance * (1 + (k * ap_num_pixels) / bg_area)
        )

    total_variance = count_rate_shot_noise_variance + bg_variance

    return ApertureCountRateAnalysis(
        total_count_rate=total_count_rate,
        median_count_rate=median_count_rate,
        mean_count_rate=mean_count_rate,
        count_rate_shot_noise_variance=count_rate_shot_noise_variance,
        bg_variance=bg_variance,
        total_count_rate_variance=total_variance,
        ap_num_pixels=ap_num_pixels,
        sigma_clip=sigma_clip,
    )


# def total_aperture_count_rate(
#     img: SwiftUVOTImage,
#     ap: Aperture,
#     background: BackgroundResult | None,
#     exposure_time_s: float,
# ) -> CountRate:
#     """
#     Takes the sum of pixels inside as the count rate, along with its error
#     If a background has not been subtracted, set background to None
#     Otherwise, assumes the background has been subtracted from 'img'
#     """
#
#     ap_stats = ApertureStats(img, ap)
#
#     total_ap_count_rate = float(ap_stats.sum)
#     aperture_area_pix = ap_stats.sum_aper_area.value
#
#     source_variance = total_ap_count_rate / exposure_time_s
#
#     # net negative count rate - estimate the error
#     if source_variance < 0.0:
#         source_variance = np.abs(source_variance)
#
#     if background is not None:
#         k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
#         bg_variance_per_pixel = background.count_rate_per_pixel.sigma**2
#         bg_area = background.bg_aperture_area
#
#         bg_variance = (
#             aperture_area_pix
#             * bg_variance_per_pixel
#             * (1 + (k * aperture_area_pix) / (exposure_time_s * bg_area))
#         )
#     else:
#         bg_variance = 0.0
#
#     total_variance = source_variance + bg_variance
#
#     return CountRate(total_ap_count_rate, sigma=np.sqrt(total_variance))


# def median_aperture_count_rate(
#     img: SwiftUVOTImage,
#     ap: Aperture,
#     background: BackgroundResult | None,
#     exposure_time_s: float,
# ) -> CountRate:
#     """
#     Takes an aperture and takes the sum of pixels inside as the count rate, along with its error
#     If a background has not been subtracted, set background to None
#     Otherwise, assumes the background has been subtracted
#     """
#
#     ap_stats = ApertureStats(img, ap)
#
#     aperture_area_pix = ap_stats.sum_aper_area.value
#
#     # estimate variance from MAD - median absolute deviation
#     mad = float(ap_stats.mad_std)
#
#     source_variance = (np.pi / 2) * mad**2 / (aperture_area_pix * exposure_time_s)
#
#     # net negative count rate - estimate the error
#     if source_variance < 0.0:
#         source_variance = np.abs(source_variance)
#
#     if background is not None:
#         k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
#         bg_variance_per_pixel = background.count_rate_per_pixel.sigma**2
#         bg_area = background.bg_aperture_area
#
#         bg_variance = (
#             aperture_area_pix
#             * bg_variance_per_pixel
#             * (1 + (k * aperture_area_pix) / (exposure_time_s * bg_area))
#         )
#     else:
#         bg_variance = 0.0
#
#     total_variance = source_variance + bg_variance
#
#     return CountRate(float(ap_stats.median), sigma=np.sqrt(total_variance))


# def median_aperture_count_rate(
#     img: SwiftUVOTImage,
#     ap: Aperture,
#     background: BackgroundResult | None,
#     exposure_time_s: float,
# ) -> CountRate:
#     """
#     Takes an aperture and takes the sum of pixels inside as the count rate, along with its error
#     If a background has not been subtracted, set background to None
#     Otherwise, assumes the background has been subtracted, and uses 'background' to factor in bg error
#     """
#
#     ap_stats = ApertureStats(img, ap)
#     # ap_stats = ApertureStats(
#     #     img, ap, sigma_clip=SigmaClip(sigma=sigma_clip, cenfunc="median")
#     # )
#
#     aperture_area_pix = ap_stats.sum_aper_area.value
#
#     source_variance = ap_stats.sum / exposure_time_s
#
#     # net negative count rate - estimate the error
#     if source_variance < 0.0:
#         source_variance = np.abs(source_variance)
#
#     if background is None:
#         bg_variance = 0.0
#     else:
#         k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
#         bg_shot_noise_variance = background.bg_shot_noise_variance
#         bg_area = background.bg_num_pixels
#
#         bg_variance = (
#             aperture_area_pix
#             * bg_shot_noise_variance
#             * (1 + (k * aperture_area_pix) / bg_area)
#         )
#
#     total_variance = source_variance + bg_variance
#
#     return CountRate(float(ap_stats.median), sigma=np.sqrt(total_variance))
