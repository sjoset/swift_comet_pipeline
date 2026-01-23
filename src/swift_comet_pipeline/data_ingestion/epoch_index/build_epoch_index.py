import numpy as np
from astropy.time import Time, TimeDelta

from swift_comet_pipeline.data_ingestion.orbit_data.find_perihelion import (
    find_perihelia,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.scp_types.compound.epoch_index import (
    EpochIndex,
    EpochIndexEntry,
)
from swift_comet_pipeline.scp_types.primitive.epoch_id import EpochID
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter


def build_epoch_index(scp: Products) -> EpochIndex | None:

    obs_log_df_unfiltered = scp.load_obs_log()
    if obs_log_df_unfiltered is None:
        return None
    obs_log_df_unfiltered = obs_log_df_unfiltered.sort_values(
        by="MID_TIME", ascending=True
    ).reset_index(drop=True)

    # obs_log_df = obs_log_df_unfiltered.copy()
    obs_log_df = obs_log_df_unfiltered[obs_log_df_unfiltered.manual_veto == False]

    epoch_index_entries = []

    for epoch_id, epoch_df in obs_log_df.groupby("epoch_id"):

        obs_time = epoch_df.MID_TIME.mean()
        epoch_length = epoch_df.MID_TIME.max() - epoch_df.MID_TIME.min()
        rh_au = epoch_df.HELIO.mean()
        helio_v_kms = epoch_df.HELIO_V.mean()
        delta_au = epoch_df.OBS_DIS.mean()
        phase_angle_deg = epoch_df.PHASE.mean()
        km_per_pix = epoch_df.KM_PER_PIX.mean()
        arcsecs_per_pix = epoch_df.ARCSECS_PER_PIXEL.mean()
        t_perihelion_list = find_perihelia(scp=scp)
        if t_perihelion_list is None:
            print("Could not find time of perihelion!")
            return None
        if len(t_perihelion_list) > 1:
            print("Found multiple perihelia during the observation window! Aborting")
            return None
        t_perihelion = t_perihelion_list[0].t_perihelion
        t_p = TimeDelta(
            (Time(np.mean(epoch_df.MID_TIME)) - t_perihelion), format="datetime"
        )
        sky_motion = epoch_df.SKY_MOTION.mean()
        sky_motion_pa = epoch_df.SKY_MOTION_PA.mean()

        exposure_times = {}
        for filter_type in UvotFilter.all_filters():
            filter_mask = epoch_df.FILTER == filter_type
            num_entries = len(epoch_df[filter_mask])
            if num_entries == 0:
                continue
            exposure_times[filter_type] = epoch_df[filter_mask].EXPOSURE.sum()  # type: ignore

        exposure_times_no_veto = {}
        for filter_type in UvotFilter.all_filters():
            ep_mask = obs_log_df_unfiltered.epoch_id == epoch_id
            filt_mask = obs_log_df_unfiltered.FILTER == filter_type
            unvetoed_mask = np.logical_and(ep_mask, filt_mask)
            unvetoed_df = obs_log_df_unfiltered[unvetoed_mask]
            num_entries = len(unvetoed_df)
            if num_entries == 0:
                continue
            exposure_times_no_veto[filter_type] = unvetoed_df.EXPOSURE.sum()

        epoch_index_entry = EpochIndexEntry(
            epoch_id=EpochID(epoch_id),
            observation_time=obs_time,
            epoch_length=epoch_length,
            rh_au=rh_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta_au,
            phase_angle_deg=phase_angle_deg,
            km_per_pix=km_per_pix,
            arcsecs_per_pix=arcsecs_per_pix,
            time_from_perihelion=t_p,
            sky_motion_arcsec_min=sky_motion,
            sky_motion_pa=sky_motion_pa,
            exposure_times=exposure_times,
            exposure_times_no_veto=exposure_times_no_veto,
        )

        epoch_index_entries.append(epoch_index_entry)

    return epoch_index_entries
