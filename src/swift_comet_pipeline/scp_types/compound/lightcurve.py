from dataclasses import dataclass, asdict
from typing import TypeAlias
from enum import StrEnum, auto

import pandas as pd

from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


class LightCurveEntrySource(StrEnum):
    from_aperture = auto()
    from_vectorial = auto()
    from_blue_spot = auto()


@dataclass(frozen=True)
class LightCurveEntry(EpochIndexEntry):
    q_h2o: float
    q_h2o_err: float
    dust_redness: DustReddeningPercent
    q_source: LightCurveEntrySource
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod
    q_oh: float = 0.0
    q_oh_err: float = 0.0


LightCurve: TypeAlias = list[LightCurveEntry | None]


def lightcurve_to_dataframe(lc: LightCurve) -> pd.DataFrame:
    """
    Takes a LightCurve and transforms each LightCurveEntry into rows of a dataframe, with columns matching the variable names in LightCurveEntry
    """
    data_dict = [asdict(lc_entry) for lc_entry in lc if lc_entry is not None]
    df = pd.DataFrame(data=data_dict)

    # json_normalize will flatten nested dicts into columns, with naming like 'eid.observation_time' as the column name
    # df = pd.json_normalize(data=data_dict)

    return df


def dataframe_to_lightcurve(df: pd.DataFrame) -> LightCurve:
    """
    Takes a dataframe with column names matching the variables in LightCurveEntry and returns a LightCurve
    """
    return df.apply(lambda row: LightCurveEntry(**row), axis=1).to_list()  # type: ignore
