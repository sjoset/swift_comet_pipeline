import astropy.units as u
from photutils.aperture import ApertureStats, CircularAperture
from photutils.aperture.stats import SigmaClip

from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def bg_sigma_clipped_aperture_stats(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
    sigma_clip: float,
) -> ApertureStats:
    """
    Calculate statistics of pixels in the image with a circular aperture at given coordinates
    Uses 3-sigma clipping.
    """
    background_aperture = CircularAperture(
        [(aperture_center.x, aperture_center.y)], r=aperture_radius
    )

    aperture_stats = ApertureStats(
        img,
        background_aperture,
        sigma_clip=SigmaClip(sigma=sigma_clip, cenfunc="median"),
    )

    return aperture_stats


def background_results_from_aperture(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
    bg_estimator: BackgroundValueEstimator,
    sigma_clip: float = 3.0,
) -> BackgroundResult:
    bg_sigma_clip_stats = bg_sigma_clipped_aperture_stats(
        img=img,
        aperture_center=aperture_center,
        aperture_radius=aperture_radius,
        sigma_clip=sigma_clip,
    )

    if bg_estimator == BackgroundValueEstimator.median:
        b_hat = bg_sigma_clip_stats.median[0]
    elif bg_estimator == BackgroundValueEstimator.mean:
        b_hat = bg_sigma_clip_stats.mean[0]

    bg_shot_noise_variance = bg_sigma_clip_stats.var[0]
    bg_num_pixels = bg_sigma_clip_stats.sum_aper_area[0].to_value(u.pix**2)  # type: ignore

    return BackgroundResult(
        b_hat=b_hat,
        bg_shot_noise_variance=bg_shot_noise_variance,
        bg_num_pixels=bg_num_pixels,
        bg_estimator=bg_estimator,
        method=BackgroundDeterminationMethod.manual_aperture_median,
        params={},
    )


# # TODO: what are we measuring here? Is it the median background +/- variance in median?
# # replace this with something else
# def bg_in_aperture(
#     img: SwiftUVOTImage,
#     aperture_center: PixelCoord,
#     aperture_radius: float,
#     bg_estimator: BackgroundValueEstimator,
#     exposure_time_s: float,
#     sigma_clip: float = 3.0,
# ) -> CountRatePerPixel:
#     bg_stats = bg_sigma_clipped_aperture_stats(
#         img=img,
#         aperture_center=aperture_center,
#         aperture_radius=aperture_radius,
#         sigma_clip=sigma_clip,
#     )
#     ap_area = np.pi * aperture_radius**2
#
#     if bg_estimator == BackgroundValueEstimator.median:
#         count_rate_per_pixel = bg_stats.median[0]
#         k = np.pi / 2
#     else:
#         count_rate_per_pixel = bg_stats.mean[0]
#         k = 1
#
#     variance = (k * bg_stats.std[0] ** 2) / (exposure_time_s * ap_area)
#
#     return CountRatePerPixel(value=count_rate_per_pixel, sigma=np.sqrt(variance))


# def bg_manual_aperture_median(
#     img: SwiftUVOTImage,
#     aperture_center: PixelCoord,
#     aperture_radius: float,
# ) -> CountRatePerPixel:
#     bg_stats = bg_sigma_clipped_aperture_stats(
#         img=img,
#         aperture_center=aperture_center,
#         aperture_radius=aperture_radius,
#     )
#
#     count_rate_per_pixel = bg_stats.median[0]
#     error_abs = np.sqrt(np.pi / 2) * bg_stats.std[0]
#
#     return CountRatePerPixel(value=count_rate_per_pixel, sigma=error_abs)


# def bg_manual_aperture_mean(
#     img: SwiftUVOTImage,
#     aperture_center: PixelCoord,
#     aperture_radius: float,
# ) -> CountRatePerPixel:
#     aperture_stats = bg_manual_aperture_stats(
#         img=img,
#         aperture_center=aperture_center,
#         aperture_radius=aperture_radius,
#     )
#
#     count_rate_per_pixel = aperture_stats.mean[0]
#
#     return CountRatePerPixel(value=count_rate_per_pixel, sigma=aperture_stats.std[0])
