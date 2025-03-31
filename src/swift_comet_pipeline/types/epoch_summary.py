from dataclasses import dataclass, asdict

import pandas as pd

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution


@dataclass
class EpochSummary:
    epoch_id: EpochID
    observation_time: pd.Timestamp
    epoch_length: pd.Timedelta
    rh_au: float
    helio_v_kms: float
    delta_au: float
    phase_angle_deg: float
    km_per_pix: float
    arcsecs_per_pix: float
    time_from_perihelion: pd.Timedelta
    pixel_resolution: SwiftPixelResolution


def dataframe_to_epoch_summary_list(
    df: pd.DataFrame,
) -> list[EpochSummary]:
    return df.apply(lambda row: EpochSummary(**row), axis=1).to_list()  # type: ignore


def epoch_summary_list_to_dataframe(
    epoch_summary_list: list[EpochSummary],
) -> pd.DataFrame:
    return pd.DataFrame(
        data=[asdict(epoch_summary) for epoch_summary in epoch_summary_list]
    )
