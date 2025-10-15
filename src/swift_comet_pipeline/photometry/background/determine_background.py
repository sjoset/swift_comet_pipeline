from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.compound.background_result import BackgroundResult
from swift_comet_pipeline.ui.mpl_ui.mpl_ui_background_manual_aperture import (
    bg_gui_manual_aperture,
)


def determine_background(
    img: SwiftUvotImage,
    exposure_map: SwiftUvotImage,
    filter_type: UvotFilter,
    epoch_id: EpochID,
    background_method: BackgroundDeterminationMethod = BackgroundDeterminationMethod.gui_manual_aperture,
) -> BackgroundResult | None:
    # if background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
    #     return bg_walking_aperture_ensemble(
    #         img=img,
    #         exposure_map=exposure_map,
    #         filter_type=filter_type,
    #         helio_r_au=epoch_summary.rh_au,
    #     )
    # elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
    #     return bg_gui_manual_aperture(img=img, filter_type=filter_type)

    if background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        return bg_gui_manual_aperture(img=img, filter_type=filter_type)
