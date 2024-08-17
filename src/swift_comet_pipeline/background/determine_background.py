from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.background.methods.bg_method_aperture_ensemble import (
    bg_walking_aperture_ensemble,
)
from swift_comet_pipeline.background.methods.bg_method_gui_manual_aperture import (
    bg_gui_manual_aperture,
)
from swift_comet_pipeline.background.methods.bg_method_swift_constant import (
    bg_swift_constant,
)
from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)


def determine_background(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    filter_type: SwiftFilter,
    background_method: BackgroundDeterminationMethod,
    helio_r_au: float,
) -> BackgroundResult:
    """The optional dict is to return any details about the method that was used"""
    if background_method == BackgroundDeterminationMethod.swift_constant:
        return bg_swift_constant(img=img, filter_type=filter_type)
    elif background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
        return bg_walking_aperture_ensemble(
            img=img,
            exposure_map=exposure_map,
            filter_type=filter_type,
            helio_r_au=helio_r_au,
        )
    elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        return bg_gui_manual_aperture(img=img, filter_type=filter_type)
