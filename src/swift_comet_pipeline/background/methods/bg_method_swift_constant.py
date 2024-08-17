from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.swift.count_rate import CountRatePerPixel
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage


def bg_swift_constant(
    img: SwiftUVOTImage, filter_type: SwiftFilter, **kwargs
) -> BackgroundResult:
    # TODO: look these up
    """Return what the background is believed to be based on the published information about the SWIFT instrumentation"""
    if filter_type == SwiftFilter.uw1:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)
    elif filter_type == SwiftFilter.uvv:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)
    else:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)

    return BackgroundResult(
        count_rate_per_pixel=count_rate_per_pixel,
        params={},
        method=BackgroundDeterminationMethod.swift_constant,
    )
