import numpy as np
from astropy.time import Time, TimeDelta

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.swift.swift_datamodes import datamode_to_pixel_resolution
from swift_comet_pipeline.types.epoch_summary import EpochSummary


def get_unstacked_epoch_summary(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> EpochSummary | None:

    unstacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    if unstacked_epoch is None:
        return None

    obs_time = unstacked_epoch.MID_TIME.mean()
    epoch_length = unstacked_epoch.MID_TIME.max() - unstacked_epoch.MID_TIME.min()
    rh_au = unstacked_epoch.HELIO.mean()
    helio_v_kms = unstacked_epoch.HELIO_V.mean()
    delta_au = unstacked_epoch.OBS_DIS.mean()
    phase_angle_deg = unstacked_epoch.PHASE.mean()
    km_per_pix = unstacked_epoch.KM_PER_PIX.mean()
    arcsecs_per_pix = unstacked_epoch.ARCSECS_PER_PIXEL.mean()
    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return None
    t_perihelion = t_perihelion_list[0].t_perihelion
    t_p = TimeDelta(
        (Time(np.mean(unstacked_epoch.MID_TIME)) - t_perihelion), format="datetime"
    )
    pixel_resolution = datamode_to_pixel_resolution(unstacked_epoch.DATAMODE[0])

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
    )


def get_epoch_summary(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> EpochSummary | None:
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    if stacked_epoch is None:
        return None

    obs_time = stacked_epoch.MID_TIME.mean()
    epoch_length = stacked_epoch.MID_TIME.max() - stacked_epoch.MID_TIME.min()
    rh_au = stacked_epoch.HELIO.mean()
    helio_v_kms = stacked_epoch.HELIO_V.mean()
    delta_au = stacked_epoch.OBS_DIS.mean()
    phase_angle_deg = stacked_epoch.PHASE.mean()
    km_per_pix = stacked_epoch.KM_PER_PIX.mean()
    arcsecs_per_pix = stacked_epoch.ARCSECS_PER_PIXEL.mean()
    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return None
    t_perihelion = t_perihelion_list[0].t_perihelion
    t_p = TimeDelta(
        (Time(np.mean(stacked_epoch.MID_TIME)) - t_perihelion), format="datetime"
    )
    pixel_resolution = datamode_to_pixel_resolution(stacked_epoch.DATAMODE[0])

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
    )
