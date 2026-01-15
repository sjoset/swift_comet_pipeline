from typing import Callable
import numpy as np
import pandas as pd
import astropy.units as u
from scipy.stats import norm


from swift_comet_pipeline.common.bayesian_expectation import (
    bayesian_expectation_over_distribution_physical_only,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ContinuumSubtractionKey,
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.post_processing.add_epoch_index_entry_to_dataframe import (
    add_epoch_index_entry_to_dataframe,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive.afrho_from_aperture_photometry import (
    dataframe_from_afrho_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive.aperture_water_production_analysis import (
    dataframe_from_aperture_water_production_analysis,
)
from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter
from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    make_hydroxyl_fragment,
)


def add_q_oh_columns(df: pd.DataFrame, oh_lifetime: float) -> pd.DataFrame:
    """
    Takes a dataframe with columns like ApertureWaterProductionAnalysisEntry and adds two columns,
    q_oh_sum and q_oh_median
    """

    new_df = df.copy()
    new_df["q_oh_sum"] = new_df.num_oh_sum / oh_lifetime
    new_df["q_oh_sum_err"] = new_df.num_oh_sum_err / oh_lifetime
    new_df["q_oh_median"] = new_df.num_oh_median / oh_lifetime
    new_df["q_oh_median_err"] = new_df.num_oh_median_err / oh_lifetime
    return new_df


def _fixed_aperture_averaged_results(
    scp: Products,
    key: ContinuumSubtractionKey,
    fixed_aperture_radius: u.Quantity,
    fixed_aperture_window_upper: u.Quantity,
    fixed_aperture_window_lower: u.Quantity,
) -> pd.DataFrame | None:
    """
    Takes the aperture water production results from the given ContinuumSubtractionKey
    and returns the same dataframe, but all columns are averaged for radial values between
    (fixed_r - window) and (fixed_r + window)

    One-row dataframe with entries like ApertureWaterProductionAnalysisEntry
    """

    ref = ProductReference(kind=ProductKind.aperture_water_production, key=key)
    if not scp.exists(ref=ref):
        print(f"_fixed_aperture_averaged_results: {key} does not exist!")
        return None

    awpa = scp.load_aperture_water_production_analysis(key=key)
    awpa_df = dataframe_from_aperture_water_production_analysis(awpa=awpa)

    fixed_aperture_radius_km = float(fixed_aperture_radius.to_value(u.km))  # type: ignore
    fixed_aperture_window_upper_km = float(fixed_aperture_window_upper.to_value(u.km))  # type: ignore
    fixed_aperture_window_lower_km = float(fixed_aperture_window_lower.to_value(u.km))  # type: ignore

    radius_mask = (
        awpa_df.aperture_r_km
        < (fixed_aperture_radius_km + fixed_aperture_window_upper_km)
    ) & (
        awpa_df.aperture_r_km
        > (fixed_aperture_radius_km - fixed_aperture_window_lower_km)
    )
    avg_over_radii_df = awpa_df.loc[radius_mask].mean(numeric_only=True).to_frame().T

    return avg_over_radii_df


def assemble_fixed_aperture_averaged_results(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    fixed_aperture_radius: u.Quantity,
    fixed_aperture_window_upper: u.Quantity,
    fixed_aperture_window_lower: u.Quantity,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns like ApertureWaterProductionAnalysisEntry, but with one row per dust redness from the epoch.
    Also computes new columns q_oh_sum and q_oh_median
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

    all_redness_results = [
        _fixed_aperture_averaged_results(
            scp=scp,
            key=x,
            fixed_aperture_radius=fixed_aperture_radius,
            fixed_aperture_window_upper=fixed_aperture_window_upper,
            fixed_aperture_window_lower=fixed_aperture_window_lower,
        )
        for x in continuum_keys
    ]
    valid_results: list[pd.DataFrame] = list(filter(lambda x: x is not None, all_redness_results))  # type: ignore
    if len(valid_results) == 0:
        print(
            f"No valid found while assembling aperture water production results for {eid.epoch_id}: {oh_filter=}, {dust_filter=}, {stacking_method=}"
        )
        return pd.DataFrame()

    valid_df = pd.concat(valid_results)

    oh_fragment = make_hydroxyl_fragment()
    scaled_oh_lifetime = oh_fragment.tau_T_s * (eid.rh_au**2)

    q_oh_added_df = add_q_oh_columns(df=valid_df, oh_lifetime=scaled_oh_lifetime)

    return q_oh_added_df


def bayesian_expectation_of_assembled_fixed_aperture_results(
    df: pd.DataFrame,
    dust_redness_mean: DustReddeningPercent,
    dust_redness_sigma: DustReddeningPercent,
    constraint_function: Callable[[pd.DataFrame], pd.Series],
    water_production_column: str = "",
) -> pd.DataFrame:
    """
    Expects a dataframe with entries like ApertureWaterProductionAnalysisEntry

    Uses constraint_function to determine whether or not points (rows) get included in the expectation values

    Adds 'dust_redness_mean', 'dust_redness_sigma', and 'percent_nonphysical' as columns

    'water_production_column' is unused in this version but old code wants to pass it in and I don't want to fix that
    """
    # TODO: remove 'water_production_column' and fix the code/notebooks that rely on passing it in

    if water_production_column != "":
        print(
            f"Warning: argument water_production_column is not used anymore - update code that passes it in to this function"
        )

    dust_prior = norm(loc=dust_redness_mean, scale=dust_redness_sigma)

    dust_redness_column = "dust_redness_pct_per_hundred_nm"
    all_other_columns = list(set(df.columns) - set(dust_redness_column))

    pbev = bayesian_expectation_over_distribution_physical_only(
        df=df,
        domain_column=dust_redness_column,
        value_columns=all_other_columns,
        pdf=dust_prior.pdf,  # type: ignore
        physical_lambda=constraint_function,
    )

    # TODO: the domain column should be the dust redness column: change it and test the marimo notebooks that rely on this function
    df = pd.DataFrame(pbev.expectations, index=[0])  # type: ignore
    # df["bayesian_expectation_domain_column"] = water_production_column
    df["bayesian_expectation_domain_column"] = dust_redness_column
    df["dust_redness_mean"] = dust_redness_mean
    df["dust_redness_sigma"] = dust_redness_sigma
    df["percent_nonphysical"] = pbev.percent_nonphysical

    return df


def assemble_afrho_fixed_aperture_results(
    scp: Products,
    eid: EpochIndexEntry,
    filter_type: UvotFilter,
    stacking_method: StackingMethod,
    fixed_aperture_radius: u.Quantity,
    fixed_aperture_window_upper: u.Quantity,
    fixed_aperture_window_lower: u.Quantity,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns like AfrhoFromAperturePhotometryAnalysisEntry, averaged around the given radius
    """

    afrho_key = EpochSubpipelineKey(
        epoch_id=eid.epoch_id, filter_type=filter_type, stacking_method=stacking_method
    )
    afapa = scp.load_afrho_from_aperture_photometry(key=afrho_key)
    afapa_df = dataframe_from_afrho_aperture_photometry_analysis(afapa=afapa)

    fixed_aperture_radius_km = float(fixed_aperture_radius.to_value(u.km))  # type: ignore
    fixed_aperture_window_upper_km = float(fixed_aperture_window_upper.to_value(u.km))  # type: ignore
    fixed_aperture_window_lower_km = float(fixed_aperture_window_lower.to_value(u.km))  # type: ignore

    # average values over the radius range specified
    radius_mask = (
        afapa_df.aperture_r_km
        < (fixed_aperture_radius_km + fixed_aperture_window_upper_km)
    ) & (
        afapa_df.aperture_r_km
        > (fixed_aperture_radius_km - fixed_aperture_window_lower_km)
    )
    avg_over_radii_df = afapa_df.loc[radius_mask].mean(numeric_only=True).to_frame().T

    avg_over_radii_df = add_epoch_index_entry_to_dataframe(
        df=avg_over_radii_df, eid=eid
    )

    avg_over_radii_df["rh_au"] = eid.rh_au * np.sign(  # type: ignore
        eid.time_from_perihelion.to_value(u.day)  # type: ignore
    )

    return avg_over_radii_df
