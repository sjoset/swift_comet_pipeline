from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.background.methods.bg_method_aperture import (
    bg_manual_aperture_mean,
    bg_manual_aperture_median,
)
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
    background_method: BackgroundDeterminationMethod,
    **kwargs,
) -> BackgroundResult:
    """The optional dict is to return any details about the method that was used"""
    if background_method == BackgroundDeterminationMethod.swift_constant:
        return bg_swift_constant(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.manual_aperture_mean:
        return bg_manual_aperture_mean(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.manual_aperture_median:
        return bg_manual_aperture_median(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
        return bg_walking_aperture_ensemble(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        return bg_gui_manual_aperture(img=img, **kwargs)
