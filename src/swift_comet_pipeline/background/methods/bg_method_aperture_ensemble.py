from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.swift.count_rate import CountRatePerPixel
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)


# TODO: add dataclass for walking ensemble input parameters?
def bg_walking_aperture_ensemble(
    img: SwiftUVOTImage,
) -> BackgroundResult:
    return BackgroundResult(
        count_rate_per_pixel=CountRatePerPixel(value=2.0, sigma=2.0),
        params={},
        method=BackgroundDeterminationMethod.walking_aperture_ensemble,
    )
