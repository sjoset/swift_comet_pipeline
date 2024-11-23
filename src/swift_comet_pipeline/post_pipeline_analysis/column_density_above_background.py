import numpy as np
import astropy.units as u

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.background_oh_production import (
    background_oh_equivalent_column_density,
)
from swift_comet_pipeline.post_pipeline_analysis.column_density_above_background_analysis import (
    ColumnDensityAboveBackgroundAnalysis,
)
from swift_comet_pipeline.post_pipeline_analysis.comet_column_density import (
    get_comet_column_density,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.uvot_image import datamode_to_pixel_resolution


def column_density_above_background(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    dust_redness: DustReddeningPercent,
    stacking_method: StackingMethod,
) -> ColumnDensityAboveBackgroundAnalysis | None:

    background_oh_cd = background_oh_equivalent_column_density(
        scp=scp,
        epoch_id=epoch_id,
        dust_redness=dust_redness,
        stacking_method=stacking_method,
    )
    if background_oh_cd is None:
        return None
    bg_oh_cd_cm2 = background_oh_cd.to_value(1 / u.cm**2)  # type: ignore

    comet_coldens = get_comet_column_density(
        scp=scp,
        epoch_id=epoch_id,
        dust_redness=dust_redness,
        stacking_method=stacking_method,
    )

    comet_cd_above_bg_mask = comet_coldens.cd_cm2 > bg_oh_cd_cm2

    comet_above_bg_rs = comet_coldens.rs_km[comet_cd_above_bg_mask] * u.km  # type: ignore
    comet_above_bg_cds = comet_coldens.cd_cm2[comet_cd_above_bg_mask] / u.cm**2  # type: ignore

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return None

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    pixel_resolution = datamode_to_pixel_resolution(stacked_epoch.DATAMODE[0])

    if len(comet_above_bg_rs) == 0:
        comet_last_usable_r = 0 * u.km  # type: ignore
        comet_last_usable_cd = 0 / u.cm**2  # type: ignore
    else:
        comet_last_usable_r = comet_above_bg_rs[-1]
        comet_last_usable_cd = comet_above_bg_cds[-1]

    num_usable_pixels = comet_last_usable_r / (km_per_pix * u.km)  # type: ignore

    return ColumnDensityAboveBackgroundAnalysis(
        epoch_id=epoch_id,
        dust_redness=dust_redness,
        stacking_method=stacking_method,
        last_usable_r=comet_last_usable_r,
        last_usable_cd=comet_last_usable_cd,
        background_oh_cd=background_oh_cd,
        num_usable_pixels_in_profile=num_usable_pixels,
        pixel_resolution=pixel_resolution,
    )
