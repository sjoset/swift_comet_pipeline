from swift_comet_pipeline.background.methods.bg_method_aperture_ensemble import (
    bg_walking_aperture_ensemble,
)
from swift_comet_pipeline.background.methods.bg_method_gui_manual_aperture import (
    bg_gui_manual_aperture,
)
from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.types.background_result import BackgroundResult
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def determine_background(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    filter_type: SwiftFilter,
    background_method: BackgroundDeterminationMethod,
    epoch_summary: EpochSummary,
) -> BackgroundResult | None:
    """The optional dict is to return any details about the method that was used"""
    if background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
        return bg_walking_aperture_ensemble(
            img=img,
            exposure_map=exposure_map,
            filter_type=filter_type,
            helio_r_au=epoch_summary.rh_au,
        )
    elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        if filter_type == SwiftFilter.uw1:
            exposure_time = epoch_summary.uw1_exposure_time_s
        elif filter_type == SwiftFilter.uvv:
            exposure_time = epoch_summary.uvv_exposure_time_s
        else:
            return None

        return bg_gui_manual_aperture(
            img=img, filter_type=filter_type, exposure_time_s=exposure_time
        )
