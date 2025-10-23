from dataclasses import dataclass
from typing import TypeAlias

import pandas as pd
import cattrs


@dataclass
class AfrhoFromAperturePhotometryAnalysisEntry:
    aperture_r_pix: float
    aperture_r_km: float
    aperture_dr_pix: float
    aperture_dr_km: float

    # magnitudes base on median profile sum and normal sum
    cumulative_median_magnitude: float
    cumulative_median_magnitude_variance: float
    cumulative_median_magnitude_err: float

    cumulative_sum_magnitude: float
    cumulative_sum_magnitude_variance: float
    cumulative_sum_magnitude_err: float

    # afrho values
    cumulative_afrho_median_cm: float
    cumulative_afrho_median_cm_var: float
    cumulative_afrho_median_cm_err: float

    cumulative_afrho_sum_cm: float
    cumulative_afrho_sum_cm_var: float
    cumulative_afrho_sum_cm_err: float

    # afrho values normalized to zero phase
    cumulative_afrho_zero_median_cm: float
    cumulative_afrho_zero_median_cm_var: float
    cumulative_afrho_zero_median_cm_err: float

    cumulative_afrho_zero_sum_cm: float
    cumulative_afrho_zero_sum_cm_var: float
    cumulative_afrho_zero_sum_cm_err: float


AfrhoFromAperturePhotometryAnalysis: TypeAlias = list[
    AfrhoFromAperturePhotometryAnalysisEntry
]


def dataframe_from_afrho_aperture_photometry_analysis(
    afapa: AfrhoFromAperturePhotometryAnalysis,
) -> pd.DataFrame:
    return pd.DataFrame(data=[cattrs.unstructure(afapae) for afapae in afapa])


def afrho_aperture_photometry_analysis_from_dataframe(
    df: pd.DataFrame,
) -> AfrhoFromAperturePhotometryAnalysis:
    afrho_dict = df.to_dict(orient="records")
    awpa = [
        cattrs.structure(
            obj=afrho_dict_entry, cl=AfrhoFromAperturePhotometryAnalysisEntry
        )
        for afrho_dict_entry in afrho_dict
    ]
    return awpa
