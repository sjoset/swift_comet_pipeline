import astropy.units as u
from photutils.aperture import ApertureStats, CircularAperture
from photutils.aperture.stats import SigmaClip

from swift_comet_pipeline.scp_types.compound.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.scp_types.primitive import *


def bg_sigma_clipped_aperture_stats(
    img: SwiftUvotImage,
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
    img: SwiftUvotImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
    bg_estimator: BackgroundValueEstimator,
    sigma_clip: float = 5.0,
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
        method=BackgroundDeterminationMethod.gui_manual_aperture,
        params={},
    )
