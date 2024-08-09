from dataclasses import dataclass, asdict
from typing import TypeAlias

import pandas as pd
from astropy.time import Time

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent


@dataclass
class LightCurveEntry:
    observation_time: Time
    time_from_perihelion_days: float
    rh_au: float

    q: float
    q_err: float

    dust_redness: DustReddeningPercent


LightCurve: TypeAlias = list[LightCurveEntry | None]


def lightcurve_to_dataframe(lc: LightCurve) -> pd.DataFrame:
    """
    Takes a LightCurve and transforms each LightCurveEntry into rows of a dataframe, with columns matching the variable names in LightCurveEntry
    """
    data_dict = [asdict(lc_entry) for lc_entry in lc if lc_entry is not None]
    df = pd.DataFrame(data=data_dict)
    return df


def dataframe_to_lightcurve(df: pd.DataFrame) -> LightCurve:
    """
    Takes a dataframe with column names matching the variables in LightCurveEntry and returns a LightCurve
    """
    return df.apply(lambda row: LightCurveEntry(**row), axis=1).to_list()
