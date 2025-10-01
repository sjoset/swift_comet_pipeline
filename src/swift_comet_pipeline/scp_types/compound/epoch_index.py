from dataclasses import dataclass, asdict

from astropy.time import TimeDelta
import pandas as pd
from pandas.core.dtypes.dtypes import NaTType

from swift_comet_pipeline.scp_types.primitive import *


@dataclass(frozen=True)
class EpochIndexEntry:
    epoch_id: EpochID

    # info about non-vetoed images
    observation_time: pd.Timestamp
    epoch_length: pd.Timedelta
    rh_au: float
    helio_v_kms: float
    delta_au: float
    phase_angle_deg: float
    km_per_pix: float
    arcsecs_per_pix: float
    time_from_perihelion: TimeDelta
    sky_motion_arcsec_min: float
    sky_motion_pa: float
    exposure_times: dict[UvotFilter, float]

    # count of entire data during, vetoed or not
    exposure_times_no_veto: dict[UvotFilter, float]


EpochIndex: TypeAlias = list[EpochIndexEntry]


def epoch_index_from_json(json_dict: dict) -> EpochIndex | None:
    # Read epoch index from file json file

    entry_list = [EpochIndexEntry(**entry) for entry in json_dict]

    typed_entries = []
    for entry in entry_list:
        observation_time = pd.Timestamp(entry.observation_time)
        epoch_length = pd.to_timedelta(entry.epoch_length)
        t_p = TimeDelta(entry.time_from_perihelion * u.day)  # type: ignore

        if isinstance(observation_time, NaTType):
            print("Could not read observation time in epoch_index_from_json")
            return None
        if isinstance(epoch_length, NaTType):
            print("Could not read epoch length in epoch_index_from_json")
            return None

        exposure_times = {}
        for filter_type, exp_t in entry.exposure_times.items():
            exposure_times[UvotFilter(filter_type)] = exp_t

        exposure_times_no_veto = {}
        for filter_type, exp_t in entry.exposure_times_no_veto.items():
            exposure_times_no_veto[UvotFilter(filter_type)] = exp_t

        new_entry = EpochIndexEntry(
            epoch_id=entry.epoch_id,
            observation_time=observation_time,
            epoch_length=epoch_length,
            rh_au=entry.rh_au,
            helio_v_kms=entry.helio_v_kms,
            delta_au=entry.delta_au,
            phase_angle_deg=entry.phase_angle_deg,
            km_per_pix=entry.km_per_pix,
            arcsecs_per_pix=entry.arcsecs_per_pix,
            time_from_perihelion=t_p,
            sky_motion_arcsec_min=entry.sky_motion_arcsec_min,
            sky_motion_pa=entry.sky_motion_pa,
            exposure_times=exposure_times,
            exposure_times_no_veto=exposure_times_no_veto,
        )
        typed_entries.append(new_entry)

    return typed_entries


def json_from_epoch_index(epoch_index: EpochIndex) -> list[dict]:
    """
    Serialize a list of EpochIndexEntry
    """
    dict_list = [asdict(epoch) for epoch in epoch_index]

    for entry_dict in dict_list:
        entry_dict["observation_time"] = entry_dict["observation_time"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        entry_dict["epoch_length"] = str(entry_dict["epoch_length"])
        entry_dict["time_from_perihelion"] = float(
            entry_dict["time_from_perihelion"].to_value(u.day)  # type: ignore
        )

    return dict_list
