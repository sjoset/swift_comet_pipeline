import numpy as np
from photutils.aperture import ApertureStats, CircularAperture

from swift_comet_pipeline.swift.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
)
from swift_comet_pipeline.swift.count_rate import CountRate, CountRatePerPixel


def aperture_count_rate(
    img: SwiftUVOTImage,
    aperture_center: PixelCoord,
    aperture_radius: float,
    bg: CountRatePerPixel,
) -> CountRate:
    """
    Constructs a circular aperture of aperture_radius at aperture_center and takes the sum of pixels
    inside as the count rate, along with its error
    """
    comet_aperture = CircularAperture(
        (aperture_center.x, aperture_center.y), r=aperture_radius
    )
    comet_aperture_stats = ApertureStats(img, comet_aperture)

    comet_count_rate = float(comet_aperture_stats.sum)
    # TODO: this is not a good error calculation but it will have to do for now
    # TODO: use calc_total_error here
    comet_count_rate_sigma = comet_aperture_stats.std

    propogated_sigma = np.sqrt(
        comet_count_rate_sigma**2
        + (comet_aperture_stats.sum_aper_area.value * bg.sigma**2)
    )

    return CountRate(value=comet_count_rate, sigma=propogated_sigma)
