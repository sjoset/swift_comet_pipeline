from photutils.aperture import ApertureStats, CircularAperture
from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.swift.count_rate import CountRatePerPixel
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage


def bg_manual_aperture_stats(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> ApertureStats:
    background_aperture = CircularAperture(
        [(aperture_x, aperture_y)], r=aperture_radius
    )

    aperture_stats = ApertureStats(img, background_aperture)

    return aperture_stats


# TODO: sigma clipping?
def bg_manual_aperture_median(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> CountRatePerPixel:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_x=aperture_x,
        aperture_y=aperture_y,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.median[0]
    # error of median is a factor larger than sigma
    error_abs = 1.2533 * aperture_stats.std[0]

    # params = {
    #     "aperture_x": float(aperture_x),
    #     "aperture_y": float(aperture_y),
    #     "aperture_radius": float(aperture_radius),
    # }

    # return BackgroundResult(
    #     count_rate_per_pixel=CountRatePerPixel(
    #         value=count_rate_per_pixel, sigma=error_abs
    #     ),
    #     params=params,
    #     method=BackgroundDeterminationMethod.manual_aperture_median,
    # )

    return CountRatePerPixel(value=count_rate_per_pixel, sigma=error_abs)


def bg_manual_aperture_mean(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> CountRatePerPixel:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_x=aperture_x,
        aperture_y=aperture_y,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.mean[0]
    error_abs = aperture_stats.std[0]

    # params = {
    #     "aperture_x": float(aperture_x),
    #     "aperture_y": float(aperture_y),
    #     "aperture_radius": float(aperture_radius),
    # }

    # return BackgroundResult(
    #     count_rate_per_pixel=CountRatePerPixel(
    #         value=count_rate_per_pixel, sigma=error_abs
    #     ),
    #     params=params,
    #     method=BackgroundDeterminationMethod.manual_aperture_mean,
    # )

    return CountRatePerPixel(value=count_rate_per_pixel, sigma=error_abs)
