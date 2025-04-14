from dataclasses import dataclass, asdict
from typing import TypeAlias

import pandas as pd
from astropy.time import Time

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent


# TODO: document these entries
@dataclass
class BayesianLightCurveEntry:
    epoch_id: EpochID
    observation_time: Time
    time_from_perihelion_days: float
    rh_au: float
    posterior_q: float
    non_detection_probability: float
    dust_mean: DustReddeningPercent
    dust_sigma: DustReddeningPercent


# mostly unused in analysis but in case we want to see internals of q interacting with dust prior
@dataclass
class WaterProductionPosteriorEntry:
    epoch_id: EpochID
    observation_time: Time
    time_from_perihelion_days: float
    rh_au: float
    q: float
    dust_redness: DustReddeningPercent
    redness_prior_prob: float
    posterior_qs: float


BayesianLightCurve: TypeAlias = list[BayesianLightCurveEntry]
WaterProductionPosteriorDataFrame: TypeAlias = pd.DataFrame


def bayesian_lightcurve_to_dataframe(blc: BayesianLightCurve) -> pd.DataFrame:
    """
    Takes a BayesianLightCurve and transforms each BayesianLightCurveEntry into rows of a dataframe, with columns matching the variable names in LightCurveEntry
    """
    data_dict = [asdict(lc_entry) for lc_entry in blc if lc_entry is not None]
    df = pd.DataFrame(data=data_dict)
    return df


def dataframe_to_bayesian_lightcurve(df: pd.DataFrame) -> BayesianLightCurve:
    """
    Takes a dataframe with column names matching the variables in BayesianLightCurveEntry and returns a BayesianLightCurve
    """
    return df.apply(lambda row: BayesianLightCurveEntry(**row), axis=1).to_list()  # type: ignore
