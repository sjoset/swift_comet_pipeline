from operator import attrgetter

import pandas as pd
from scipy.stats import norm

from swift_comet_pipeline.common.bayesian_expectation import (
    bayesian_expectation_over_distribution,
)
from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.photometry.dust.reddening_translate import (
    recalculate_redness_with_new_filter_pair,
)
from swift_comet_pipeline.post_processing.add_epoch_index_entry_to_dataframe import (
    add_epoch_index_entry_to_dataframe,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ContinuumSubtractionKey,
    Products,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.compound.radial_profile_water_production import (
    RadialProfileWaterProductionAnalysis,
)
from swift_comet_pipeline.swift.filters.filter_wavelengths import (
    effective_wavelength_of_filter_observing_solar_flux,
)


def dataframe_from_radial_water_production_analysis(
    rpwpa: RadialProfileWaterProductionAnalysis | None,
    dust_redness: DustReddeningPercent,
) -> pd.DataFrame | None:
    """
    Transforms a RadialProfileWaterProductionAnalysis into a dataframe with columns holding the
    near, far, full fits along with the measured oh column density, all at the given redness
    """

    if rpwpa is None:
        return None

    # start the dataframe and add the redness value
    df = pd.DataFrame(
        {"oh_column_density": rpwpa.oh_column_density}, index=pd.Series([0])
    )
    df["dust_redness_pct_per_hundred_nm"] = dust_redness

    # for each type of fit, add the associated VectorialModelFits to a column named after the fit type
    for ft in VectorialFitType.all_types():
        df[ft] = getattr(rpwpa, ft)

    return df


def assemble_vectorial_model_results(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
) -> pd.DataFrame:
    """
    For the the given epoch and filters, gathers Q(H2O) derived from vectorial model fitting of the OH column density at every redness:
        near-nucleus, far region, and full oh column density curve
    The columns will be dataclasses of ColumnDensity and VectorialModelFits for every dust redness
    """

    continuum_keys = [
        ContinuumSubtractionKey(
            epoch_id=eid.epoch_id,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            dust_redness_pct_per_hundred_nm=x,
        )
        for x in scp.dust_rednesses
    ]

    # load fitting results for every redness
    all_redness_results = [
        scp.load_radial_profile_water_production_analysis(key=x) for x in continuum_keys
    ]
    # break the list of results into list of one-row dataframes, one per redness
    all_redness_result_dfs = [
        dataframe_from_radial_water_production_analysis(
            rpwpa=x, dust_redness=k.dust_redness_pct_per_hundred_nm
        )
        for x, k in zip(all_redness_results, continuum_keys)
    ]
    # if calculations for a certain redness are missing, remove them from the list
    valid_results: list[pd.DataFrame] = list(filter(lambda x: x is not None, all_redness_result_dfs))  # type: ignore
    if len(valid_results) == 0:
        print(
            f"No valid results found while assembling vectorial water production results for {eid.epoch_id}: {oh_filter=}, {dust_filter=}, {stacking_method=}"
        )
        return pd.DataFrame()

    # combine the one-row dataframes into a final dataframe and attach epoch info
    valid_df = pd.concat(valid_results).reset_index(drop=True)
    return add_epoch_index_entry_to_dataframe(df=valid_df, eid=eid)


def assemble_vectorial_model_production_rates(
    df: pd.DataFrame, oh_filter: UvotFilter, dust_filter: UvotFilter
) -> pd.DataFrame:
    """
    Using the dataframe from assemble_vectorial_model_results, build a new dataframe that lists:
        Q(H2O), Q error, Q relative error, redness in uw1/uvv, effective redness at actual filter pair, beta parameter for dust
    so that we have water production rates as a function of redness in the 'standard' uw1/uvv pair and effective redness

    Assumes the columns produced by assemble_vectorial_model_results are present, and that dust_redness_pct_per_hundred_nm is in terms of uw1/uvv mid wavelength
    """

    # break the dataframe into near, far, and full curve fitting Q values for each redness, including the relative error for each fit

    # this 'dust_redness_pct_per_hundred_nm' is in terms of uw1/uvv pair
    q_df: pd.DataFrame = df[["dust_redness_pct_per_hundred_nm"]].copy()  # type: ignore
    q_df["near_fit_q"] = df.near_fit.map(attrgetter("best_fit_q_per_s"))
    q_df["near_fit_q_err"] = df.near_fit.map(attrgetter("best_fit_q_per_s_err"))
    q_df["near_fit_q_rel_err"] = q_df.near_fit_q_err / q_df.near_fit_q

    q_df["far_fit_q"] = df.far_fit.map(attrgetter("best_fit_q_per_s"))
    q_df["far_fit_q_err"] = df.far_fit.map(attrgetter("best_fit_q_per_s_err"))
    q_df["far_fit_q_rel_err"] = q_df.far_fit_q_err / q_df.far_fit_q

    q_df["full_fit_q"] = df.full_fit.map(attrgetter("best_fit_q_per_s"))
    q_df["full_fit_q_err"] = df.full_fit.map(attrgetter("best_fit_q_per_s_err"))
    q_df["full_fit_q_rel_err"] = q_df.full_fit_q_err / q_df.full_fit_q

    uw1_uvv_mid_wave = (
        effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uvv)
        + effective_wavelength_of_filter_observing_solar_flux(UvotFilter.uw1)
    ) / 2
    uw1_uvv_mid_nm = float(uw1_uvv_mid_wave.to_value(u.nm))  # type: ignore

    rrwnfp = np.vectorize(recalculate_redness_with_new_filter_pair)
    q_df["effective_redness"] = rrwnfp(
        known_redness=q_df.dust_redness_pct_per_hundred_nm.to_numpy(),
        old_mid_wave_nm=uw1_uvv_mid_nm,
        new_filter_one=oh_filter,
        new_filter_two=dust_filter,
    )

    # TODO: should we calculate beta for uw1/uvv pair, and rename the column 'beta_parameter' to 'effective_beta_parameter'?

    bp = np.vectorize(beta_parameter)
    q_df["beta_parameter"] = bp(
        q_df.effective_redness.to_numpy(),
        oh_filter=oh_filter,
        dust_filter=dust_filter,
    )

    return q_df


def assemble_vectorial_model_bayesian_production_rates(
    df: pd.DataFrame,
    dust_redness_sigma: DustReddeningPercent,
    calculate_for_dust_rednesses: list[DustReddeningPercent] | None = None,
) -> pd.DataFrame:
    """
    Take a dataframe produced by assemble_vectorial_model_production_rates, and produces a dataframe with additional columns 'bayes_'* that are the result of using our prior
    at each redness in calculate_for_dust_rednesses.
    If calculate_for_dust_rednesses is 'None', then we calculate for every redness in the dataframe.

    At each redness S', build a gaussian that peaks at S' with sigma = dust_redness_sigma and produce bayes_far_fit_q, bayes_far_fit_q_err, etc.

    We really only care about the far fitting results at this point in the calculations, so we omit the near/full curve fit bayesian calculations
    """

    dfc = df.copy()

    if calculate_for_dust_rednesses is None:
        dust_rednesses = df.dust_redness_pct_per_hundred_nm.to_numpy()
    else:
        dust_rednesses = calculate_for_dust_rednesses

    # list of columns for analysis
    vectorial_source_columns = ["far_fit_q", "far_fit_q_err", "far_fit_q_rel_err"]
    # where to store each result
    bayes_destination_columns = ["bayes_" + x for x in vectorial_source_columns]

    vec_to_bayes_dict = {
        v: b for v, b in zip(vectorial_source_columns, bayes_destination_columns)
    }

    # column checking
    for c in bayes_destination_columns:
        # initialize the column with np.nan if the column doesn't exist: we might not be asked to calculate whole column,
        # so mark everything with np.nan and overwrite later if we do calculate that row
        if c not in dfc.columns:
            dfc[c] = np.nan

    for dust_redness in dust_rednesses:
        # build Bayesian prior
        dust_prior = norm(loc=dust_redness, scale=dust_redness_sigma)

        # do the calculation for each of our vectorial columns
        bayes_result_list = bayesian_expectation_over_distribution(
            df=dfc,
            domain_column="dust_redness_pct_per_hundred_nm",
            pdf=dust_prior.pdf,  # type: ignore
            value_columns=vectorial_source_columns,
        )

        # fill in the dataframe with the results
        for bayes_result in bayes_result_list:
            dest_col = vec_to_bayes_dict[bayes_result.value_column]
            dfc.loc[dfc.dust_redness_pct_per_hundred_nm == dust_redness, dest_col] = (
                bayes_result.expectation_value
            )

    return dfc
