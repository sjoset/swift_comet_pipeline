from typing import Callable
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_processing.bayesian_expectation import (
    bayesian_expectation_over_distribution,
)
from swift_comet_pipeline.post_processing.vectorial_fitting_reliability import (
    do_vectorial_fitting_reliability_post_processing,
)
from swift_comet_pipeline.types.bayesian_expectation import (
    BayesianExpectationResultFromDataframe,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.lightcurve import LightCurve, dataframe_to_lightcurve
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


def get_epoch_vectorial_fitting_reliability_expectation(
    scp: SwiftCometPipeline,
    swift_project_config: SwiftProjectConfig,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    dust_prior_pdf: Callable,
    dust_rednesses: list[DustReddeningPercent],
) -> BayesianExpectationResultFromDataframe:
    df = do_vectorial_fitting_reliability_post_processing(
        scp=scp,
        stacking_method=stacking_method,
        epoch_id=epoch_id,
        dust_rednesses=dust_rednesses,
        vectorial_fitting_requires_km=swift_project_config.vectorial_fitting_requires_km,
        num_psfs_required=3,
    )
    assert df is not None
    df["vfr"] = df.vectorial_fitting_reliable.astype(float)

    es = bayesian_expectation_over_distribution(
        df=df, domain_column="dust_redness", value_columns=["vfr"], pdf=dust_prior_pdf
    )

    return es[0]


def build_unified_lightcurve(
    swift_project_config: SwiftProjectConfig,
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    dust_prior_mean: DustReddeningPercent,
    dust_prior_sigma: DustReddeningPercent,
    vectorial_fit_source: PipelineFilesEnum,
) -> LightCurve | None:

    # get bayesian aperture results
    bayes_df_raw = scp.get_product_data(
        pf=PipelineFilesEnum.bayesian_aperture_lightcurve,
        stacking_method=stacking_method,
    )
    if bayes_df_raw is None:
        print("Could not find bayesian aperture analysis!")
        return None

    bayes_aperture_df = bayes_df_raw[
        (bayes_df_raw.dust_mean == dust_prior_mean)
        & (bayes_df_raw.dust_sigma == dust_prior_sigma)
    ]
    bayes_aperture_df = bayes_aperture_df.reset_index(drop=True).set_index("epoch_id")
    bayes_aperture_df.insert(loc=1, column="q", value=bayes_aperture_df.posterior_q)

    # get vectorial results
    lc_df = scp.get_product_data(
        pf=vectorial_fit_source, stacking_method=stacking_method
    )
    if lc_df is None:
        print("Could not load vectorial fitting data!")
        return None
    lc_df = lc_df.reset_index(drop=True).set_index("epoch_id")

    if vectorial_fit_source == PipelineFilesEnum.best_near_fit_vectorial_lightcurve:
        lc_df = lc_df.rename(columns={"near_fit_q": "q", "near_fit_q_err": "q_err"})
    elif vectorial_fit_source == PipelineFilesEnum.best_far_fit_vectorial_lightcurve:
        lc_df = lc_df.rename(columns={"far_fit_q": "q", "far_fit_q_err": "q_err"})
    elif vectorial_fit_source == PipelineFilesEnum.best_full_fit_vectorial_lightcurve:
        lc_df = lc_df.rename(columns={"full_fit_q": "q", "full_fit_q_err": "q_err"})
    else:
        print("vectorial_fit_source error while building unified lightcurve!")
        return None

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(-100.0, 100.0, num=201, endpoint=True)
    ]

    # do all of the post-processing for vectorial reliability here
    for epoch_id in tqdm(epoch_ids):
        do_vectorial_fitting_reliability_post_processing(
            scp=scp,
            stacking_method=stacking_method,
            epoch_id=epoch_id,
            dust_rednesses=dust_rednesses,
            vectorial_fitting_requires_km=swift_project_config.vectorial_fitting_requires_km,
            num_psfs_required=3,  # TODO: magic number
        )

    # build the gaussian pdf we use for bayesian expectation, with the rednesses we are evaluating over
    dust_prior = norm(loc=float(dust_prior_mean), scale=dust_prior_sigma)

    # build dataframe about whether vectorial fitting is reliable enough to be used
    vfr_dict = {
        x: get_epoch_vectorial_fitting_reliability_expectation(
            swift_project_config=swift_project_config,
            scp=scp,
            epoch_id=x,
            stacking_method=stacking_method,
            dust_prior_pdf=dust_prior.pdf,
            dust_rednesses=dust_rednesses,
        )
        for x in epoch_ids
    }
    vfr_df = pd.DataFrame.from_dict(
        {k: asdict(v) for k, v in vfr_dict.items()}, orient="index"
    )
    vfr_df.index.name = "epoch_id"
    vfr_df["use_vectorial"] = vfr_df.expectation_value > 0.5

    # decide between vectorial or bayes
    unified_lc_df = lc_df.where(vfr_df.use_vectorial, bayes_aperture_df)

    # we have to reset the index of unified_lc_df because using epoch_id as index
    # removes it from the list of columns, which we use to pack into the LightCurve constructor
    # build and return results
    lc: LightCurve = dataframe_to_lightcurve(df=unified_lc_df.reset_index())

    return lc
