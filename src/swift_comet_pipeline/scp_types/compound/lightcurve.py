from dataclasses import dataclass, asdict
from typing import TypeAlias
from enum import StrEnum, auto

import pandas as pd

# from astropy.time import Time

from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


class LightCurveEntrySource(StrEnum):
    from_aperture = auto()
    from_vectorial = auto()


@dataclass(frozen=True)
class LightCurveEntry(EpochIndexEntry):
    q: float
    q_err: float
    dust_redness: DustReddeningPercent
    q_source: LightCurveEntrySource
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod


# @dataclass
# class LightCurveEntry:
#     eid: EpochIndexEntry
#     q: float
#     q_err: float
#     dust_redness: DustReddeningPercent


# class LightCurveEntry:
#     epoch_id: EpochID
#     observation_time: Time
#     time_from_perihelion_days: float
#     rh_au: float
#
#     q: float
#     q_err: float
#
#     dust_redness: DustReddeningPercent


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


# def lightcurve_to_dataframe(lc: LightCurve) -> pd.DataFrame:
#     """
#     Takes a LightCurve and transforms each LightCurveEntry into rows of a dataframe, with columns matching the variable names in LightCurveEntry
#     """
#     data_dict = [asdict(lc_entry) for lc_entry in lc if lc_entry is not None]
#     df = pd.DataFrame(data=data_dict)
#     return df
#
#
# def dataframe_to_lightcurve(df: pd.DataFrame) -> LightCurve:
#     """
#     Takes a dataframe with column names matching the variables in LightCurveEntry and returns a LightCurve
#     """
#     return df.apply(lambda row: LightCurveEntry(**row), axis=1).to_list()  # type: ignore

# def radial_profile_water_production_analysis_from_json(
#     json_dict: dict,
# ) -> RadialProfileWaterProductionAnalysis:
#     return from_unstructured(obj=json_dict, c=RadialProfileWaterProductionAnalysis)
#
#
# def json_from_radial_profile_water_production_analysis(
#     rpwpa: RadialProfileWaterProductionAnalysis,
# ) -> dict:
#     return to_unstructured(obj=rpwpa)
