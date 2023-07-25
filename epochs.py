import pathlib
import calendar
import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.time import Time

from typing import TypeAlias

from swift_types import SwiftObservationLog
from observation_log import read_observation_log, write_observation_log

__all__ = [
    "Epoch",
    "file_name_from_epoch",
    "epoch_schema",
    "read_epoch",
    "write_epoch",
    "epoch_from_obs_log",
]


Epoch: TypeAlias = pd.DataFrame


def file_name_from_epoch(epoch: Epoch) -> str:
    epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
    day = epoch_start.day
    month = calendar.month_abbr[epoch_start.month]
    year = epoch_start.year

    return f"{year}_{day:02d}_{month}"


# TODO: merge this schema into observation log and just initialize all these possible error values in build_observation_log
def epoch_schema() -> pa.lib.Schema:
    epoch_schema = pa.schema(
        [
            pa.field("manual_veto", pa.bool_()),
        ]
    )

    return epoch_schema


def read_epoch(epoch_path: pathlib.Path) -> Epoch:
    """
    Allow read_observation_log to do post-load processing on SwiftObservationLog columns
    """
    epoch = read_observation_log(epoch_path, additional_schema=epoch_schema())

    return epoch


def write_epoch(epoch: Epoch, epoch_path: pathlib.Path) -> None:
    # do any column processing of our own here

    write_observation_log(epoch, epoch_path, additional_schema=epoch_schema())


def epoch_from_obs_log(obs_log: SwiftObservationLog) -> Epoch:
    """
    Adds default values to columns from an observation log so that it matches epoch_schema
    """
    obs_log["manual_veto"] = False

    return obs_log
