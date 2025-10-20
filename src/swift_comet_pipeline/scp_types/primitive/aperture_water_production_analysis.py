from dataclasses import dataclass, asdict
from typing import TypeAlias

import cattrs
import pandas as pd


@dataclass
class ApertureWaterProductionAnalysisEntry:
    aperture_r_pix: float
    aperture_r_km: float
    aperture_dr_pix: float
    aperture_dr_km: float

    dust_redness_pct_per_hundred_nm: float

    oh_counts_sum: float
    oh_counts_sum_variance: float
    oh_counts_sum_err: float

    oh_counts_median: float
    oh_counts_median_variance: float
    oh_counts_median_err: float

    oh_flux_sum: float
    oh_flux_sum_variance: float
    oh_flux_sum_err: float

    oh_flux_median: float
    oh_flux_median_variance: float
    oh_flux_median_err: float

    num_oh_sum: float
    num_oh_sum_variance: float
    num_oh_sum_err: float

    num_oh_median: float
    num_oh_median_variance: float
    num_oh_median_err: float

    # match the vectorial model number of fragments in this aperture to what we measure
    aperture_matched_q_h2o_sum: float
    aperture_matched_q_h2o_sum_variance: float
    aperture_matched_q_h2o_sum_err: float
    aperture_matched_q_h2o_median: float
    aperture_matched_q_h2o_median_variance: float
    aperture_matched_q_h2o_median_err: float

    # take the observed OH number to be the total produced by a vectorial model, and find the corresponding production
    equivalent_q_h2o_sum: float
    equivalent_q_h2o_sum_variance: float
    equivalent_q_h2o_sum_err: float
    equivalent_q_h2o_median: float
    equivalent_q_h2o_median_variance: float
    equivalent_q_h2o_median_err: float


ApertureWaterProductionAnalysis: TypeAlias = list[ApertureWaterProductionAnalysisEntry]


def dataframe_from_aperture_water_production_analysis(
    awpa: ApertureWaterProductionAnalysis,
) -> pd.DataFrame:
    return pd.DataFrame(data=[asdict(awpae) for awpae in awpa])


def aperture_water_production_analysis_from_dataframe(
    df: pd.DataFrame,
) -> ApertureWaterProductionAnalysis:
    water_dict = df.to_dict(orient="records")
    awpa = [
        cattrs.structure(obj=water_dict_entry, cl=ApertureWaterProductionAnalysisEntry)
        for water_dict_entry in water_dict
    ]
    return awpa
