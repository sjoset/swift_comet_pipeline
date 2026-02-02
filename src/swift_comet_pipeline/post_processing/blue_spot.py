from functools import partial
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import norm

from swift_comet_pipeline.common.bayesian_expectation import (
    bayesian_expectation_over_distribution,
)
from swift_comet_pipeline.scp_types.primitive import *


def calculate_blue_spot_from_column_densities(
    vectorial_oh_cds: np.ndarray, data_derived_oh_cds: np.ndarray, rs_km: np.ndarray
) -> float:
    """
    Takes the column densities of the vectorial predictions 'vectorial_oh_cds' on the radial grid rs_km,
    along with the actual column density derived from data 'data_derived_oh_cds' and calculates the excess
    number of OH molecules beyond the vectorial model prediction
    Column densities are expected to be floats in units of 1/cm**2
    """

    # TODO: re-sample along radius and linearly interpolate both column densities
    # NOTE: don't filter for only positive contributions to excess OH
    bs_cds_raw = data_derived_oh_cds - vectorial_oh_cds

    # check if we even have enough data to do this
    if len(rs_km) < 2:
        return 0.0

    rs_cm = rs_km * 1e5
    bs_cds_times_rho = bs_cds_raw * rs_cm
    dr_cm = rs_cm[1] - rs_cm[0]

    oh_excess = 2 * np.pi * trapezoid(bs_cds_times_rho, dx=dr_cm)
    return oh_excess


def compute_blue_spot_num_oh_single_row(
    row: pd.Series,
    blue_spot_extent_km: float,
    # savgol_window_length: int,
    # savgol_window_polyorder: int,
) -> float:
    """
    Expects a row of a dataframe with columns like that produced by assemble_vectorial_model_results()

    Computes the number of excess OH molecules near the nucleus.
    """

    fit_rs_all = row.far_fit.vectorial_column_density.rs_km
    data_rs_all = row.oh_column_density.rs_km
    assert np.sum(fit_rs_all - data_rs_all) == 0

    fit_cds_all = row.far_fit.vectorial_column_density.cd_cm2
    data_cds_all = row.oh_column_density.cd_cm2
    # data_cds_smoothed_all = savgol_filter(data_cds_all, window_length=savgol_window_length, polyorder=savgol_window_polyorder)

    _bs_r_mask = data_rs_all < blue_spot_extent_km
    # _bs_cd_mask = data_cds_smoothed_all > 0
    _bs_cd_mask = data_cds_all > 0
    _bs_total_mask = (_bs_r_mask) & (_bs_cd_mask)

    fit_cds = fit_cds_all[_bs_total_mask]
    data_cds = data_cds_all[_bs_total_mask]
    # data_cds_smoothed = savgol_filter(data_cds, window_length=savgol_window_length, polyorder=savgol_window_polyorder)
    fit_rs = fit_rs_all[_bs_total_mask]

    num_oh_bs = calculate_blue_spot_from_column_densities(
        vectorial_oh_cds=fit_cds, data_derived_oh_cds=data_cds, rs_km=fit_rs
    )
    return num_oh_bs


def blue_spot_df_from_vectorial_df(
    df: pd.DataFrame, scaled_oh_lifetime: float, blue_spot_extent_km: float
) -> pd.DataFrame:
    """
    Takes a dataframe like that produced by assemble_vectorial_model_results(), strips some columns, and adds
    blue spot OH count and blue spot OH production rate
    This dataframe will be a function of dust redness
    """

    blue_spot_func = partial(
        compute_blue_spot_num_oh_single_row, blue_spot_extent_km=blue_spot_extent_km
    )

    bsdf = df.copy()
    bsdf["blue_spot_num_oh"] = bsdf.apply(blue_spot_func, axis=1)
    bsdf["blue_spot_q_oh"] = bsdf.blue_spot_num_oh / scaled_oh_lifetime

    bsdf = bsdf.drop(columns=["far_fit", "near_fit", "full_fit", "oh_column_density"])
    return bsdf


def bayes_blue_spot_df_from_blue_spot_df(
    df: pd.DataFrame,
    dust_redness_sigma: DustReddeningPercent,
    calculate_for_dust_rednesses: list[DustReddeningPercent] | None = None,
) -> pd.DataFrame:
    """
    Takes result of blue_spot_df_from_vectorial_df and adds columns for bayesian calculation of blue spot OH number and OH production rate

    Results placed into 'bayes_'* for blue_spot_num_oh and blue_spot_q_oh columns
    """

    bbsdf = df.copy()

    if calculate_for_dust_rednesses is None:
        dust_rednesses = df.dust_redness_pct_per_hundred_nm.to_numpy()
    else:
        dust_rednesses = calculate_for_dust_rednesses

    blue_spot_source_columns = ["blue_spot_num_oh", "blue_spot_q_oh"]
    bayes_destination_columns = ["bayes_" + x for x in blue_spot_source_columns]
    blue_spot_to_bayes_dict = {
        bs: bayes
        for bs, bayes in zip(blue_spot_source_columns, bayes_destination_columns)
    }

    # initialize columns to np.nan if they aren't yet present
    for c in bayes_destination_columns:
        if not c in bbsdf.columns:
            bbsdf[c] = np.nan

    for dust_redness in dust_rednesses:
        dust_prior = norm(loc=dust_redness, scale=dust_redness_sigma)
        # compute the values for each column - returned in a list
        bayes_result_list = bayesian_expectation_over_distribution(
            df=df,
            domain_column="dust_redness_pct_per_hundred_nm",
            value_columns=blue_spot_source_columns,
            pdf=dust_prior.pdf,  # type: ignore
        )

        for bayes_result in bayes_result_list:
            dest_col = blue_spot_to_bayes_dict[bayes_result.value_column]
            bbsdf.loc[
                bbsdf.dust_redness_pct_per_hundred_nm == dust_redness, dest_col
            ] = bayes_result.expectation_value

    return bbsdf
