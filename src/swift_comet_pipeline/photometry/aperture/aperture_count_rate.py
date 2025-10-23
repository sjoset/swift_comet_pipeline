import logging as log

import numpy as np
from photutils.aperture import Aperture, ApertureStats
from photutils.aperture.stats import SigmaClip

from swift_comet_pipeline.scp_types.compound.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.scp_types.primitive import *


# TODO: clean up old code
def bg_variance_in_aperture(aperture_area: float, bg_result: BackgroundResult) -> float:
    """
    Aperture area in pixels^2: this should be the effective area of the aperture if noise is correlated
    """

    k = 1 if bg_result.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
    bg_shot_noise_variance = bg_result.bg_shot_noise_variance
    bg_area = bg_result.bg_num_pixels

    bg_shot_noise_variance_in_aperture = aperture_area * bg_shot_noise_variance
    bg_estimation_variance = aperture_area**2 * k * bg_shot_noise_variance / bg_area

    bg_variance = bg_shot_noise_variance_in_aperture + bg_estimation_variance

    return bg_variance


def count_rate_variance_in_aperture(
    total_count_rate: float, exposure_time_s: float
) -> float:

    source_shot_noise_variance = total_count_rate / exposure_time_s

    # net negative count rate - estimate the error
    if source_shot_noise_variance < 0.0:
        log.info(
            "Negative count rate results in negative poisson variance - estimating shot noise"
        )
        source_shot_noise_variance = np.abs(source_shot_noise_variance)

    return source_shot_noise_variance


# def total_variance_in_aperture(
#     total_count_rate: float, exposure_time_s: float, bg_variance_in_ap: float
# ) -> float:
#
#     source_shot_noise_variance = total_count_rate / exposure_time_s
#
#     # net negative count rate - estimate the error
#     if source_shot_noise_variance < 0.0:
#         log.info(
#             "Negative count rate results in negative poisson variance - estimating shot noise"
#         )
#         source_shot_noise_variance = np.abs(source_shot_noise_variance)
#
#     total_variance = source_shot_noise_variance + bg_variance_in_ap
#
#     return total_variance


def aperture_analysis(
    img: SwiftUvotImage,
    ap: Aperture,
    background: BackgroundResult | None,
    exposure_time_s: float,
    sigma_clip: float = 3.0,
) -> ApertureCountRateAnalysis:
    """
    Takes an aperture and returns ApertureCountRateAnalysis - the sum, median, and mean signal in the aperture, along with errors

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

    # effective area of aperture due to correlated pixels
    # TODO: calculate this to better handle correlated errors
    ap_eff_num_pixels: float = ap_num_pixels

    # # variances
    # count_rate_shot_noise_variance: float = ap_stats.sum / exposure_time_s  # type: ignore
    #
    # # net negative count rate - estimate the error
    # if count_rate_shot_noise_variance < 0.0:
    #     log.info(
    #         "Negative count rate results in negative poisson variance - estimating shot noise"
    #     )
    #     count_rate_shot_noise_variance = np.abs(count_rate_shot_noise_variance)

    if background is None:
        bg_variance = 0.0
    else:
        # k = 1 if background.bg_estimator == BackgroundValueEstimator.mean else np.pi / 2
        # bg_shot_noise_variance = background.bg_shot_noise_variance
        # bg_area = background.bg_num_pixels
        #
        # bg_shot_noise_variance_in_aperture = ap_eff_num_pixels * bg_shot_noise_variance
        # bg_estimation_variance = ap_num_pixels**2 * k * bg_shot_noise_variance / bg_area
        #
        # bg_variance = bg_shot_noise_variance_in_aperture + bg_estimation_variance
        bg_variance = bg_variance_in_aperture(
            aperture_area=ap_eff_num_pixels, bg_result=background
        )

    # total_variance = count_rate_shot_noise_variance + bg_variance

    # total_variance = total_variance_in_aperture(total_count_rate=ap_stats.sum, exposure_time_s=exposure_time_s, bg_variance_in_ap=bg_variance)

    count_rate_shot_noise_variance = count_rate_variance_in_aperture(
        total_count_rate=ap_stats.sum, exposure_time_s=exposure_time_s  # type: ignore
    )

    total_variance = count_rate_shot_noise_variance + bg_variance

    return ApertureCountRateAnalysis(
        sum_count_rate=total_count_rate,
        median_count_rate=median_count_rate,
        mean_count_rate=mean_count_rate,
        count_rate_shot_noise_variance=count_rate_shot_noise_variance,
        bg_variance=bg_variance,
        sum_count_rate_variance=total_variance,
        ap_num_pixels=ap_num_pixels,
        sigma_clip=sigma_clip,
    )
