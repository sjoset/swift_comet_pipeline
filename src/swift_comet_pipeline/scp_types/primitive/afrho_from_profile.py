from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd
import cattrs


@dataclass
class AfrhoFromRadialProfileEntry:
    r_pix: float
    count_rate: float
    r_km: float

    # annuli for the profile samples
    inner_r_pix: float
    outer_r_pix: float
    annulus_area_pix: float

    annulus_count_rate: float
    annulus_count_rate_var: float
    annulus_count_rate_err: float

    # # total
    cumulative_count_rate: float
    cumulative_count_rate_var: float
    cumulative_count_rate_err: float

    cumulative_magnitude: float
    cumulative_magnitude_var: float
    cumulative_magnitude_err: float

    cumulative_afrho_cm: float
    cumulative_afrho_cm_var: float
    cumulative_afrho_cm_err: float

    cumulative_afrho_zero_cm: float
    cumulative_afrho_zero_cm_var: float
    cumulative_afrho_zero_cm_err: float


AfrhoFromRadialProfile: TypeAlias = list[AfrhoFromRadialProfileEntry]


def dataframe_from_afrho_from_radial_profile(
    afrp: AfrhoFromRadialProfile,
) -> pd.DataFrame:
    return pd.DataFrame(data=[cattrs.unstructure(afrpe) for afrpe in afrp])


def afrho_from_radial_profile_from_dataframe(
    df: pd.DataFrame,
) -> AfrhoFromRadialProfile:
    afrho_dict = df.to_dict(orient="records")
    afrp = [
        cattrs.structure(obj=afrho_dict_entry, cl=AfrhoFromRadialProfileEntry)
        for afrho_dict_entry in afrho_dict
    ]
    return afrp
