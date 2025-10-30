from dataclasses import dataclass


# TODO: remove if unused
@dataclass
class ApertureActiveAreaAnalysisEntry:
    """
    These will be computed with ApertureWaterProductionAnalysisEntry
    """

    aperture_matched_q_h2o_sum_upper_limit_active_area_km2: float
    aperture_matched_q_h2o_sum_upper_limit_active_area_km2_var: float
    aperture_matched_q_h2o_sum_upper_limit_active_area_km2_err: float

    aperture_matched_q_h2o_median_upper_limit_active_area_km2: float
    aperture_matched_q_h2o_median_upper_limit_active_area_km2_var: float
    aperture_matched_q_h2o_median_upper_limit_active_area_km2_err: float

    equivalent_q_h2o_sum_upper_limit_active_area_km2: float
    equivalent_q_h2o_sum_upper_limit_active_area_km2_var: float
    equivalent_q_h2o_sum_upper_limit_active_area_km2_err: float

    equivalent_q_h2o_median_upper_limit_active_area_km2: float
    equivalent_q_h2o_median_upper_limit_active_area_km2_var: float
    equivalent_q_h2o_median_upper_limit_active_area_km2_err: float

    aperture_matched_q_h2o_sum_lower_limit_active_area_km2: float
    aperture_matched_q_h2o_sum_lower_limit_active_area_km2_var: float
    aperture_matched_q_h2o_sum_lower_limit_active_area_km2_err: float

    aperture_matched_q_h2o_median_lower_limit_active_area_km2: float
    aperture_matched_q_h2o_median_lower_limit_active_area_km2_var: float
    aperture_matched_q_h2o_median_lower_limit_active_area_km2_err: float

    equivalent_q_h2o_sum_lower_limit_active_area_km2: float
    equivalent_q_h2o_sum_lower_limit_active_area_km2_var: float
    equivalent_q_h2o_sum_lower_limit_active_area_km2_err: float

    equivalent_q_h2o_median_lower_limit_active_area_km2: float
    equivalent_q_h2o_median_lower_limit_active_area_km2_var: float
    equivalent_q_h2o_median_lower_limit_active_area_km2_err: float
