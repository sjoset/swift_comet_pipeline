from functools import cache

import numpy as np
from astropy.time import Time, TimeDelta

from swift_comet_pipeline.observationlog.epoch_typing import Epoch, EpochID
from swift_comet_pipeline.observationlog.stacked_epoch import StackedEpoch
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.swift.swift_datamodes import datamode_to_pixel_resolution
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
    pixel_resolution = datamode_to_pixel_resolution(epoch.DATAMODE[0])
    uw1_mask = epoch.FILTER == SwiftFilter.uw1
    uvv_mask = epoch.FILTER == SwiftFilter.uvv
    uw1_exposure_time = epoch[uw1_mask].EXPOSURE.sum()
    uvv_exposure_time = epoch[uvv_mask].EXPOSURE.sum()

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
        pixel_resolution=pixel_resolution,
        uw1_exposure_time_s=uw1_exposure_time,
        uvv_exposure_time_s=uvv_exposure_time,
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
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return None

    return make_epoch_summary(scp=scp, epoch_id=epoch_id, epoch=stacked_epoch)
