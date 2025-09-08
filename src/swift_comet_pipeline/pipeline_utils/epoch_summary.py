from functools import cache

import numpy as np
from astropy.time import Time, TimeDelta

from swift_comet_pipeline.observationlog.epoch_typing import Epoch, EpochID
from swift_comet_pipeline.observationlog.stacked_epoch import StackedEpoch
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.swift_filter import SwiftFilter


def make_epoch_summary(
    scp: SwiftCometPipeline, epoch_id: EpochID, epoch: Epoch | StackedEpoch
) -> EpochSummary | None:

    obs_time = epoch.MID_TIME.mean()
    epoch_length = epoch.MID_TIME.max() - epoch.MID_TIME.min()
    rh_au = epoch.HELIO.mean()
    helio_v_kms = epoch.HELIO_V.mean()
    delta_au = epoch.OBS_DIS.mean()
    phase_angle_deg = epoch.PHASE.mean()
    km_per_pix = epoch.KM_PER_PIX.mean()
    arcsecs_per_pix = epoch.ARCSECS_PER_PIXEL.mean()
    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return None
    t_perihelion = t_perihelion_list[0].t_perihelion
    t_p = TimeDelta((Time(np.mean(epoch.MID_TIME)) - t_perihelion), format="datetime")
    uw1_mask = epoch.FILTER == SwiftFilter.uw1
    uvv_mask = epoch.FILTER == SwiftFilter.uvv
    uw1_exposure_time = epoch[uw1_mask].EXPOSURE.sum()
    uvv_exposure_time = epoch[uvv_mask].EXPOSURE.sum()
    sky_motion = epoch.SKY_MOTION.mean()
    sky_motion_pa = epoch.SKY_MOTION_PA.mean()

    return EpochSummary(
        epoch_id=epoch_id,
        observation_time=obs_time,
        epoch_length=epoch_length,
        rh_au=rh_au,
        helio_v_kms=helio_v_kms,
        delta_au=delta_au,
        phase_angle_deg=phase_angle_deg,
        km_per_pix=km_per_pix,
        arcsecs_per_pix=arcsecs_per_pix,
        time_from_perihelion=t_p,
        uw1_exposure_time_s=uw1_exposure_time,
        uvv_exposure_time_s=uvv_exposure_time,
        sky_motion_arcsec_min=sky_motion,
        sky_motion_pa=sky_motion_pa,
    )


@cache
def get_unstacked_epoch_summary(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> EpochSummary | None:

    unstacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    if unstacked_epoch is None:
        return None

    return make_epoch_summary(scp=scp, epoch_id=epoch_id, epoch=unstacked_epoch)


@cache
def get_epoch_summary(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> EpochSummary | None:
    """
    This fixes the km_per_pix to the highest value found in the epoch, because after stacking all images should be scaled to 1 arcesecond per pixel.
    If we use the mean value, we have a mixture of event-mode plate scales and data-mode plate scales - everything should be the same scale after stacking!
    """
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return None

    es = make_epoch_summary(scp=scp, epoch_id=epoch_id, epoch=stacked_epoch)
    if es is not None:
        # TODO: log that we are fixing this value
        es.km_per_pix = np.max(stacked_epoch.KM_PER_PIX)

    return es
