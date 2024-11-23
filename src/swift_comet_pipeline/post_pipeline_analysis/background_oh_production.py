import numpy as np
import astropy.units as u

from swift_comet_pipeline.background.background_result import (
    yaml_dict_to_background_result,
)
from swift_comet_pipeline.comet.calculate_column_density import (
    surface_brightness_profile_to_column_density,
)
from swift_comet_pipeline.comet.comet_surface_brightness_profile import (
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import datamode_to_pixel_resolution
from swift_comet_pipeline.water_production.flux_OH import beta_parameter


def background_oh_equivalent_column_density(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    dust_redness: DustReddeningPercent,
    stacking_method: StackingMethod,
) -> u.Quantity | None:
    beta = beta_parameter(dust_redness=dust_redness)

    bg_uw1 = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    if bg_uw1 is None:
        return None
    bg_uw1 = yaml_dict_to_background_result(raw_yaml=bg_uw1)

    bg_uvv = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if bg_uvv is None:
        return None
    bg_uvv = yaml_dict_to_background_result(raw_yaml=bg_uvv)

    # image after bg subtraction has normal noise distribution with sigma - make new count rate with 1-sigma as signal level
    bg_oh_cr = (
        bg_uw1.count_rate_per_pixel.sigma - beta * bg_uvv.count_rate_per_pixel.sigma
    )

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return None

    pixel_resolution = datamode_to_pixel_resolution(stacked_epoch.DATAMODE[0])
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    helio_v = np.mean(stacked_epoch.HELIO_V) * (u.km / u.s)  # type: ignore
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

    countrate_profile = np.array([bg_oh_cr])

    bg_oh_surf_brightness = countrate_profile_to_surface_brightness(
        countrate_profile=countrate_profile,
        delta=delta,
        pixel_resolution=pixel_resolution,
    )

    bg_oh_cd = surface_brightness_profile_to_column_density(
        surface_brightness_profile=bg_oh_surf_brightness,
        delta=delta,
        helio_r=helio_r,
        helio_v=helio_v,
    )

    return bg_oh_cd[0]
