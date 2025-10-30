from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from swift_comet_pipeline.common.parallel_compute import parallel_compute_float
from swift_comet_pipeline.scp_types.primitive.dataframe_column_and_error_set import (
    DataframeColumnAndErrorSet,
)


# TODO: rewrite this to use ApertureActiveAnalysisEntry?
# TODO: decide if we drop this entirely


@dataclass
class _ActiveAreaDataframeCalculation:
    """
    q describes columns holding a floating point value for the water production rate in molecules/sec
    active_area describes the columns that will hold the result for active area
    aa_func is a function that takes a water production rate as a float in molecules/second and returns an area in km**2 as
    a float
    """

    q: DataframeColumnAndErrorSet
    active_area: DataframeColumnAndErrorSet
    aa_func: Callable


def active_area_dataframe_calculation(
    df: pd.DataFrame, aadc: _ActiveAreaDataframeCalculation
) -> None:
    aa_func_vectorized = np.vectorize(aadc.aa_func)

    # compute the active area with the production rate
    qs = df[aadc.q.col].to_numpy(copy=False)
    active_areas_km2 = parallel_compute_float(aadc.aa_func, qs, do_tqdm=True)
    # active_areas_km2 = aa_func_vectorized(qs)
    df[aadc.active_area.col] = active_areas_km2

    # compute the active area using q+one sigma and q-one sigma, and take the average error of these two
    # as the stochastic error for the active area calculation
    q_errs = df[aadc.q.col_err].to_numpy(copy=False)
    one_sigma_above_qs = qs + q_errs
    one_sigma_below_qs = qs - q_errs
    one_sigma_above = aa_func_vectorized(one_sigma_above_qs)
    one_sigma_below = aa_func_vectorized(one_sigma_below_qs)
    df[aadc.active_area.col_err] = (
        np.abs(active_areas_km2 - one_sigma_above)
        + np.abs(active_areas_km2 - one_sigma_below)
    ) / 2
    df[aadc.active_area.col_var] = df[aadc.active_area.col_err] ** 2


# def prepare_df(water_df):
#     water_production_columns = [
#         "aperture_matched_q_h2o_sum",
#         "aperture_matched_q_h2o_median",
#         "equivalent_q_h2o_sum",
#         "equivalent_q_h2o_median",
#     ]
#     tqdm.write("Starting active area calculations...")
#     # active area estimates
#     active_area_upper_limit_func = partial(
#         estimate_active_area_km2, rh_au=eid.rh_au, sub_solar_latitude_deg=0.0
#     )
#     active_area_lower_limit_func = partial(
#         estimate_active_area_km2, rh_au=eid.rh_au, sub_solar_latitude_deg=90.0
#     )
#
#     for (prefix, aa_func), q_column in product(
#         [
#             ("_lower", active_area_lower_limit_func),
#             ("_upper", active_area_upper_limit_func),
#         ],
#         water_production_columns,
#     ):
#         qcolset = _DataframeColumnAndErrorSet(
#             col=q_column, col_var=q_column + "_var", col_err=q_column + "_err"
#         )
#         aa_col = q_column + prefix + "_limit_active_area_km2"
#         aa_col_var = aa_col + "_var"
#         aa_col_err = aa_col + "_err"
#         aa_colset = _DataframeColumnAndErrorSet(
#             col=aa_col, col_var=aa_col_var, col_err=aa_col_err
#         )
#         aadc = _ActiveAreaDataframeCalculation(
#             q=qcolset, active_area=aa_colset, aa_func=aa_func
#         )
#
#         tqdm.write(f"{aadc.q.col} --> {aadc.active_area.col} ...")
#         _active_area_calculation(df=water_df, aadc=aadc)
#     tqdm.write("Complete")
