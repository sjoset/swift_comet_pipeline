import numpy as np
import astropy.units as u

from swift_comet_pipeline.comet.calculate_column_density import (
    surface_brightness_profile_to_column_density,
)
from swift_comet_pipeline.comet.countrate_profile_to_surface_brightness import (
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.comet_column_density import (
    get_comet_column_density_from_extracted_profile,
)
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.pipeline_utils.get_uw1_and_uvv import (
    get_uw1_and_uvv_background_results,
)
from swift_comet_pipeline.swift.swift_datamodes import datamode_to_pixel_resolution
from swift_comet_pipeline.types.column_density_above_background_analysis import (
    ColumnDensityAboveBackgroundAnalysis,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.uw1_uvv_pair import uw1uvv_getter


# TODO: this file probably belongs in background/
def background_oh_equivalent_column_density(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    dust_redness: DustReddeningPercent,
    stacking_method: StackingMethod,
    sigma_level: float = 3.0,
) -> u.Quantity | None:
    """
    Calculates the average background count rate of OH (which should be zero) along with its 1-sigma uncertainty.
    The 1-sigma uncertainty is then extended upward to sigma_level, and that count rate is used to derive a corresponding OH column density.
    Then any column density we measure above this level is outside of our background range.
    """
    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    beta = beta_parameter(dust_redness=dust_redness)

    bgs = get_uw1_and_uvv_background_results(
        scp=scp, epoch_id=epoch_id, stacking_method=stacking_method
    )
    assert bgs is not None
    bg_uw1, bg_uvv = uw1uvv_getter(bgs)

    bg_oh_cr = bg_uw1.count_rate_per_pixel - beta * bg_uvv.count_rate_per_pixel
    bg_oh_cr_err = bg_oh_cr.sigma * sigma_level

    countrate_profile = np.array([bg_oh_cr_err])
    bg_oh_surf_brightness = countrate_profile_to_surface_brightness(
        countrate_profile=countrate_profile,
        epoch_summary=epoch_summary,
    )

    bg_oh_cd = surface_brightness_profile_to_column_density(
        surface_brightness_profile=bg_oh_surf_brightness, epoch_summary=epoch_summary
    )

    return bg_oh_cd[0]


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

    comet_coldens = get_comet_column_density_from_extracted_profile(
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
