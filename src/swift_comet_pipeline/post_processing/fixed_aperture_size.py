from typing import Callable
import pandas as pd
import astropy.units as u
from scipy.stats import norm


from swift_comet_pipeline.common.bayesian_expectation import (
    bayesian_expectation_over_distribution_physical_only,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ContinuumSubtractionKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
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


# def fixed_aperture_results_old(
#     scp: Products,
#     key: ContinuumSubtractionKey,
#     fixed_aperture_radius_km: float,
#     fixed_aperture_window_upper_km: float,
#     fixed_aperture_window_lower_km: float,
# ) -> pd.DataFrame:
#     """
#     Takes the aperture water production results from the given ContinuumSubtractionKey
#     and returns the same dataframe, but limited to radial distances between
#     (fixed_r - window) and (fixed_r + window)
#     """
#
#     awpa = scp.load_aperture_water_production_analysis(key=key)
#     awpa_df = dataframe_from_aperture_water_production_analysis(awpa=awpa)
#
#     df = awpa_df[
#         (
#             awpa_df.aperture_r_km
#             < (fixed_aperture_radius_km + fixed_aperture_window_upper_km)
#         )
#         & (
#             awpa_df.aperture_r_km
#             > (fixed_aperture_radius_km - fixed_aperture_window_lower_km)
#         )
#     ]
#
#     assert isinstance(df, pd.DataFrame)
#     return df


def add_q_oh_columns(df: pd.DataFrame, oh_lifetime: float) -> pd.DataFrame:
    """
    Takes a dataframe with columns like ApertureWaterProductionAnalysisEntry and adds two columns,
    q_oh_sum and q_oh_median
    """

    new_df = df.copy()
    new_df["q_oh_sum"] = new_df.num_oh_sum / oh_lifetime
    new_df["q_oh_median"] = new_df.num_oh_median / oh_lifetime
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
    water_production_column: str,
    dust_redness_mean: DustReddeningPercent,
    dust_redness_sigma: DustReddeningPercent,
    constraint_function: Callable[[pd.DataFrame], pd.Series],
):
    """
    Expects a dataframe with entries like ApertureWaterProductionAnalysisEntry

    Takes the expectation value of the columns while only using data with rows where 'water_production_column' is positive
    """

    dust_prior = norm(loc=dust_redness_mean, scale=dust_redness_sigma)

    dust_redness_column = "dust_redness_pct_per_hundred_nm"
    all_other_columns = list(set(df.columns) - set(dust_redness_column))

    pbev = bayesian_expectation_over_distribution_physical_only(
        df=df,
        domain_column=dust_redness_column,
        value_columns=all_other_columns,
        pdf=dust_prior.pdf,  # type: ignore
        physical_lambda=constraint_function,
        # physical_lambda=lambda x: x[water_production_column] > 0,
    )

    df = pd.DataFrame(pbev.expectations, index=[0])  # type: ignore
    df["bayesian_expectation_domain_column"] = water_production_column
    df["dust_redness_mean"] = dust_redness_mean
    df["dust_redness_sigma"] = dust_redness_sigma
    df["percent_nonphysical"] = pbev.percent_nonphysical

    return df
