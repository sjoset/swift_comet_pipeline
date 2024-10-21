from photutils.aperture import ApertureStats, CircularAperture
from photutils.aperture.stats import SigmaClip
from swift_comet_pipeline.swift.count_rate import CountRatePerPixel
from swift_comet_pipeline.swift.uvot_image import PixelCoord, SwiftUVOTImage


def bg_manual_aperture_stats(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
) -> ApertureStats:
    """
    Calculate statistics of pixels in the image with a circular aperture at given coordinates
    Uses 3-sigma clipping.
    """
    background_aperture = CircularAperture(
        [(aperture_center.x, aperture_center.y)], r=aperture_radius
    )

    aperture_stats = ApertureStats(
        img, background_aperture, sigma_clip=SigmaClip(sigma=3.0, cenfunc="median")
    )

    return aperture_stats


def bg_manual_aperture_median(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
) -> CountRatePerPixel:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_center=aperture_center,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.median[0]
    # error of median is a factor larger than sigma
    error_abs = 1.2533 * aperture_stats.std[0]

    return CountRatePerPixel(value=count_rate_per_pixel, sigma=error_abs)


def bg_manual_aperture_mean(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
) -> CountRatePerPixel:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_center=aperture_center,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.mean[0]
    error_abs = aperture_stats.std[0]

    return CountRatePerPixel(value=count_rate_per_pixel, sigma=error_abs)
