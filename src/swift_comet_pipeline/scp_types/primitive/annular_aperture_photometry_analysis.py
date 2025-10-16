from dataclasses import dataclass, asdict
from typing import TypeAlias

import pandas as pd
import cattrs


@dataclass
class AnnularAperturePhotometryAnalysisEntry:
    aperture_r_pix: float
    aperture_r_km: float
    aperture_dr_pix: float
    aperture_dr_km: float
    cumulative_aperture_area: float

    # median count rate times the aperture area, at each annulus
    area_scaled_median: float
    area_scaled_median_variance: float
    area_scaled_median_err: float

    # cumulative median count rate from r=0 to aperture_r_pix
    cumulative_area_scaled_median: float
    cumulative_area_scaled_median_variance: float
    cumulative_area_scaled_median_err: float

    # cumulative sum of aperture count rates from r=0 to aperture_r_pix
    cumulative_sum: float
    cumulative_sum_variance: float
    cumulative_sum_err: float

    # magnitudes base on median profile sum and normal sum
    cumulative_median_magnitude: float
    cumulative_median_magnitude_variance: float
    cumulative_median_magnitude_err: float
    cumulative_sum_magnitude: float
    cumulative_sum_magnitude_variance: float
    cumulative_sum_magnitude_err: float


AnnularAperturePhotometryAnalysis: TypeAlias = list[
    AnnularAperturePhotometryAnalysisEntry
]


def annular_aperture_photometry_analysis_from_dataframe(
    df: pd.DataFrame,
) -> AnnularAperturePhotometryAnalysis:
    df_row_dict = df.to_dict(orient="records")
    aapa = [
        cattrs.structure(df_row, AnnularAperturePhotometryAnalysisEntry)
        for df_row in df_row_dict
    ]
    return aapa


def dataframe_from_annular_aperture_photometry_analysis(
    aapa: AnnularAperturePhotometryAnalysis,
) -> pd.DataFrame:
    return pd.DataFrame(data=[asdict(aapae) for aapae in aapa])
