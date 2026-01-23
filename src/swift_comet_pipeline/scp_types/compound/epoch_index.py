from dataclasses import dataclass

from astropy.time import TimeDelta
import pandas as pd

from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.serialization.structure_unstructure import (
    from_unstructured,
    to_unstructured,
)


# TODO: add position angles for velocity/dust and sun/ion tail instead of looking them up later
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
    """
    Get EpochIndex from json stored in a dictionary for serialization
    Sorts entries by epoch_id
    """
    ei = from_unstructured(obj=json_dict, c=EpochIndex)
    ei_sorted = sorted(ei, key=lambda x: x.epoch_id)
    return ei_sorted


def json_from_epoch_index(epoch_index: EpochIndex) -> list[dict]:
    dict_list = [
        to_unstructured(epoch_index_entry) for epoch_index_entry in epoch_index
    ]
    return dict_list
