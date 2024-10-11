from dataclasses import dataclass, asdict
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy.stats import norm
from astropy.time import Time

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import (
    LightCurve,
    lightcurve_to_dataframe,
)


@dataclass
class BayesianLightCurveEntry:
    observation_time: Time
    time_from_perihelion_days: float
    rh_au: float
    posterior_q: float
    non_detection_probability: float
    dust_mean: DustReddeningPercent
    dust_sigma: DustReddeningPercent


# mostly unused in analysis but in case we want to see internals of q interacting with dust prior
@dataclass
class WaterProductionPosteriorEntry:
    dust_redness: DustReddeningPercent
    time_from_perihelion_days: float
    rh_au: float
    observation_time: Time
    q: float
    redness_prior_prob: float
    posterior_qs: float


BayesianLightCurve: TypeAlias = list[BayesianLightCurveEntry]
WaterProductionPosteriorDataFrame: TypeAlias = pd.DataFrame


def make_gaussian_prior(mean_reddening: DustReddeningPercent, sigma_reddening: float):
    return norm(loc=float(mean_reddening), scale=sigma_reddening)


def bayesian_lightcurve_from_aperture_lightcurve(
    lc: LightCurve, mean_reddening: DustReddeningPercent, sigma_reddening: float
) -> tuple[BayesianLightCurve, WaterProductionPosteriorDataFrame]:
    # TODO: document function and the columns of WaterProductionPosteriorDataFrame

    # Assume a gaussian prior for the dust redness, and a uniform likelihood associated with q - we don't have any belief about what q should pop out of our model

    lc_df = lightcurve_to_dataframe(lc=lc)

    # there may be multiple plateaus found for a given observation time and dust redness - which means multiple q(h2o).
    # this collapses the lightcurve into having just one q(h2o) per time/redness pair by averaging the plateau's q(h2o)s.
    averaged_lc_df = (
        lc_df.groupby(
            ["time_from_perihelion_days", "dust_redness", "rh_au", "observation_time"]
        )[["q"]]
        .mean()
        .reset_index()
    )

    # try to guess which dust rednesses the pipeline used in its analysis
    all_rednesses = sorted(averaged_lc_df.dust_redness.unique())

    # Entries for dust rednesses are missing if they came out to be a non-detection of water,
    # so fill in the rows with the missing redness value and a q(h2o) of zero
    filled_in_df_list = []
    for _, sub_df in averaged_lc_df.groupby(["time_from_perihelion_days"]):
        temp_df = (
            sub_df.set_index("dust_redness").reindex(all_rednesses).reset_index().copy()
        )
        # missing dust rednesses came out to be a non-detection of production, so fill that in
        temp_df[["q"]] = temp_df[["q"]].fillna(0.0)
        # fill in the missing times by using the existing valid values: these should all be the same because we grouped by them
        temp_df[["time_from_perihelion_days"]] = (
            temp_df[["time_from_perihelion_days"]].ffill().bfill()
        )
        # similarly for rh_au and observation_time
        temp_df[["rh_au"]] = temp_df[["rh_au"]].ffill().bfill()
        temp_df[["observation_time"]] = temp_df[["observation_time"]].ffill().bfill()

        filled_in_df_list.append(temp_df)

    filled_in_df = pd.concat(filled_in_df_list)

    # make our dust redness prior
    gaussian_prior = make_gaussian_prior(
        mean_reddening=mean_reddening, sigma_reddening=sigma_reddening
    )

    # fill in dataframe with prior probabilities
    filled_in_df["redness_prior_prob"] = filled_in_df["dust_redness"].map(
        lambda x: gaussian_prior.pdf(x)  # type: ignore
    )

    # calculate the water production times the prior redness probability
    filled_in_df["posterior_qs"] = filled_in_df.q * filled_in_df.redness_prior_prob

    posterior_q_list = []
    non_detection_probs = []
    for _, sub_df in filled_in_df.groupby("time_from_perihelion_days"):
        non_zero_prod_mask = sub_df.q != 0.0
        # Expectation value of q, with sum taken over the redness
        posterior_q = np.sum(sub_df.posterior_qs[non_zero_prod_mask])
        posterior_q_list.append(posterior_q)
        non_detection_prob = np.sum(sub_df.redness_prior_prob[~non_zero_prod_mask])
        non_detection_probs.append(non_detection_prob)

    result_df = pd.DataFrame(
        {
            "time_from_perihelion_days": filled_in_df.time_from_perihelion_days.unique(),
            "posterior_q": posterior_q_list,
            "rh_au": filled_in_df.rh_au.unique(),
            "observation_time": filled_in_df.observation_time.unique(),
            "non_detection_probability": non_detection_probs,
            "dust_mean": mean_reddening,
            "dust_sigma": sigma_reddening,
        }
    )

    # tag the heliocentric distance with plus or minus, negative distance being pre-perihelion
    result_df.rh_au = result_df.rh_au * np.sign(result_df.time_from_perihelion_days)

    return (dataframe_to_bayesian_lightcurve(df=result_df), filled_in_df)


def bayesian_lightcurve_to_dataframe(blc: BayesianLightCurve) -> pd.DataFrame:
    """
    Takes a BayesianLightCurve and transforms each BayesianLightCurveEntry into rows of a dataframe, with columns matching the variable names in LightCurveEntry
    """
    data_dict = [asdict(lc_entry) for lc_entry in blc if lc_entry is not None]
    df = pd.DataFrame(data=data_dict)
    return df


def dataframe_to_bayesian_lightcurve(df: pd.DataFrame) -> BayesianLightCurve:
    """
    Takes a dataframe with column names matching the variables in BayesianLightCurveEntry and returns a BayesianLightCurve
    """
    return df.apply(lambda row: BayesianLightCurveEntry(**row), axis=1).to_list()
