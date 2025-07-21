import time
from itertools import product

import numpy as np
import pandas as pd
import astropy.units as u
from tqdm import tqdm

from swift_comet_pipeline.dust.dust_limits import (
    get_dust_redness_lower_limit,
    get_dust_redness_upper_limit,
)
from swift_comet_pipeline.dust.dust_redness_prior import (
    get_dust_redness_mean_prior,
    get_dust_redness_sigma_prior,
)
from swift_comet_pipeline.lightcurve.lightcurve_aperture import (
    lightcurve_from_aperture_plateaus,
)
from swift_comet_pipeline.lightcurve.lightcurve_bayesian import (
    bayesian_lightcurve_from_aperture_lightcurve,
)
from swift_comet_pipeline.lightcurve.lightcurve_vectorial import (
    lightcurve_from_vectorial_fits,
)
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_processing.unified_lightcurve import (
    build_unified_lightcurve,
)
from swift_comet_pipeline.tui.tui_common import get_selection
from swift_comet_pipeline.types.bayesian_lightcurve import (
    bayesian_lightcurve_to_dataframe,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.lightcurve import (
    dataframe_to_lightcurve,
    lightcurve_to_dataframe,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig
from swift_comet_pipeline.types.vectorial_model_fit_type import VectorialFitType


def find_minimum_percent_error_in_lightcurve(
    df: pd.DataFrame, production_column_name: str, production_err_column_name: str
) -> pd.DataFrame:
    # TODO: document this function and its inputs

    positive_production_mask = df.q > 0.0
    # positive_redness_mask = df.dust_redness > 0.0

    by_epoch = df[positive_production_mask].groupby("time_from_perihelion_days")

    min_error_rows = []
    for _, epoch_df in by_epoch:
        idx_min = np.argmin(
            epoch_df[production_err_column_name] / epoch_df[production_column_name]
        )
        # the double brackets below for [[idx_min]] forces pandas to return the row as a DataFrame and not Series, allowing
        # us to concat them and keep our column structures
        min_error_rows.append(epoch_df.iloc[[idx_min]])

    return pd.concat(min_error_rows)


def build_lightcurves_step(swift_project_config: SwiftProjectConfig) -> None:

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    build_aperture_lightcurves_step(
        swift_project_config=swift_project_config, stacking_method=stacking_method
    )

    build_aperture_lightcurves_bayesian_step(
        swift_project_config=swift_project_config, stacking_method=stacking_method
    )

    build_vectorial_lightcurves_step(
        swift_project_config=swift_project_config, stacking_method=stacking_method
    )

    build_unified_lightcurve_step(
        swift_project_config=swift_project_config, stacking_method=stacking_method
    )


def build_aperture_lightcurves_step(
    swift_project_config: SwiftProjectConfig, stacking_method: StackingMethod
) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    if scp.exists(
        pf=PipelineFilesEnum.aperture_lightcurve, stacking_method=stacking_method
    ):
        print("Aperture lightcurve already exists, skipping.")
        return

    # aperture productions
    aperture_lc = lightcurve_from_aperture_plateaus(
        scp=scp,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
    )

    if aperture_lc is None:
        print("Building aperture lightcurve failed!")
        return

    print("Writing aperture lightcurve results...")
    lc_prod = scp.get_product(
        pf=PipelineFilesEnum.aperture_lightcurve, stacking_method=stacking_method
    )
    assert lc_prod is not None
    lc_prod.data = lightcurve_to_dataframe(lc=aperture_lc)
    lc_prod.write()


def build_aperture_lightcurves_bayesian_step(
    swift_project_config: SwiftProjectConfig, stacking_method: StackingMethod
) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    if scp.exists(
        pf=PipelineFilesEnum.bayesian_aperture_lightcurve,
        stacking_method=stacking_method,
    ):
        print("Bayesian aperture lightcurve already exists, skipping.")
        return

    aperture_lc = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_lightcurve, stacking_method=stacking_method
    )
    assert aperture_lc is not None
    aperture_lc = dataframe_to_lightcurve(df=aperture_lc)

    # TODO: magic numbers
    dust_upper = get_dust_redness_upper_limit()
    dust_lower = get_dust_redness_lower_limit()
    dust_mean_reddenings = np.linspace(
        dust_lower,
        dust_upper,
        num=int(np.round(dust_upper - dust_lower + 1)),
        endpoint=True,
    )
    dust_sigma_reddenings = [1.0, 5.0, 10.0, 20.0]

    print("Calculating results for bayesian aperture lightcurve analysis...")
    bayes_lc_dfs = []
    for dust_mean, dust_sigma in product(dust_mean_reddenings, dust_sigma_reddenings):
        blc, _ = bayesian_lightcurve_from_aperture_lightcurve(
            lc=aperture_lc, mean_reddening=dust_mean, sigma_reddening=dust_sigma
        )
        bayes_lc_dfs.append(bayesian_lightcurve_to_dataframe(blc=blc))

    bayes_df = pd.concat(bayes_lc_dfs, axis=0)

    print("Writing results...")
    bayes_product = scp.get_product(
        pf=PipelineFilesEnum.bayesian_aperture_lightcurve,
        stacking_method=stacking_method,
    )
    assert bayes_product is not None
    bayes_product.data = bayes_df
    bayes_product.write()


def build_vectorial_lightcurves_step(
    swift_project_config: SwiftProjectConfig, stacking_method: StackingMethod
) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    vectorial_fit_product_types = [
        PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
        PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
        PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
        PipelineFilesEnum.complete_vectorial_lightcurve,
    ]
    if all(
        [
            scp.exists(pf=vfpt, stacking_method=stacking_method)
            for vfpt in vectorial_fit_product_types
        ]
    ):
        print("All vectorial lightcurves exist, skipping.")
        return

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    near_far_radius = swift_project_config.near_far_split_radius_km * u.km  # type: ignore
    print(f"Using near/far vectorial fitting split of {near_far_radius}...")

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(-100.0, 100.0, num=201, endpoint=True)
    ]

    precompute_t_start = time.perf_counter()
    print("Precomputing vectorial models...")
    _ = lightcurve_from_vectorial_fits(
        scp=scp,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
        dust_redness=dust_rednesses[0],
        fit_type=VectorialFitType.near_fit,
        near_far_radius=near_far_radius,
    )
    precompute_t_end = time.perf_counter()
    print(f"All models done in {precompute_t_end - precompute_t_start:.2f} seconds")

    print("Starting lightcurve calculations...")
    all_fit_start = time.perf_counter()
    fitted_vectorial_lcs = {}
    fit_types_and_rednesses = list(
        product(VectorialFitType.all_types(), dust_rednesses)
    )
    dust_progress_bar = tqdm(
        fit_types_and_rednesses, total=len(fit_types_and_rednesses)
    )
    for fitting_kind, dust_redness in dust_progress_bar:
        dust_progress_bar.set_description(
            f"Fit type: {fitting_kind.value}, Dust redness: {dust_redness}"
        )
        fitted_vectorial_lcs[fitting_kind, dust_redness] = (
            lightcurve_from_vectorial_fits(
                scp=scp,
                stacking_method=stacking_method,
                t_perihelion=t_perihelion,
                dust_redness=dust_redness,
                fit_type=fitting_kind,
                near_far_radius=near_far_radius,
            )
        )
    all_fit_end = time.perf_counter()
    print(f"All fits done in {all_fit_end - all_fit_start:.2f} seconds")

    fit_dfs = {}
    best_fit_dfs = {}
    for fit_type in VectorialFitType.all_types():
        fit_dfs[fit_type] = pd.concat(
            [
                lightcurve_to_dataframe(lc=fitted_vectorial_lcs[fit_type, x])
                for x in dust_rednesses
            ]
        )
        best_fit_dfs[fit_type] = find_minimum_percent_error_in_lightcurve(
            df=fit_dfs[fit_type],
            production_column_name="q",
            production_err_column_name="q_err",
        )

    vectorial_fitting_pairs = [
        (
            VectorialFitType.near_fit,
            PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
        ),
        (VectorialFitType.far_fit, PipelineFilesEnum.best_far_fit_vectorial_lightcurve),
        (
            VectorialFitType.full_fit,
            PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
        ),
    ]

    # write out the vectorial best-fit lightcurves for each fitting type
    for fit_type, product_type in vectorial_fitting_pairs:
        p = scp.get_product(pf=product_type, stacking_method=stacking_method)
        assert p is not None
        p.data = best_fit_dfs[fit_type]
        p.write()

    # full vectorial fitting data for each fit type: rename columns to reflect their fitting type,
    # and we can combine them into one dataframe for the 'complete' lightcurve below
    renaming_pairs = [
        (VectorialFitType.near_fit, "near_fit_q", "near_fit_q_err"),
        (VectorialFitType.far_fit, "far_fit_q", "far_fit_q_err"),
        (VectorialFitType.full_fit, "full_fit_q", "full_fit_q_err"),
    ]
    for fit_type, q_col, q_err_col in renaming_pairs:
        fit_dfs[fit_type] = fit_dfs[fit_type].rename(
            columns={"q": q_col, "q_err": q_err_col}
        )

    # tag heliocentric distance with a negative sign before perihelion
    for fit_type in VectorialFitType.all_types():
        fit_dfs[fit_type].rh_au = fit_dfs[fit_type].rh_au * np.sign(
            fit_dfs[fit_type].time_from_perihelion_days
        )

    # drop the duplicated columns: each df has its own time_from_perihelion and observation_time columns
    # complete_vectorial_lightcurves_df = pd.concat([df_near, df_far, df_full], axis=1)
    complete_vectorial_lightcurves_df = pd.concat(
        [fit_dfs[fit_type] for fit_type in VectorialFitType.all_types()], axis=1
    )
    complete_vectorial_lightcurves_df = complete_vectorial_lightcurves_df.loc[
        :, ~complete_vectorial_lightcurves_df.columns.duplicated()
    ]

    lc_complete = scp.get_product(
        pf=PipelineFilesEnum.complete_vectorial_lightcurve,
        stacking_method=stacking_method,
    )
    assert lc_complete is not None
    lc_complete.data = complete_vectorial_lightcurves_df
    lc_complete.write()


def build_unified_lightcurve_step(
    swift_project_config: SwiftProjectConfig, stacking_method: StackingMethod
) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    print("TODO: get bayesian options for dust redness")
    print("Building unified lightcurve...")
    vectorial_fit_source = PipelineFilesEnum.best_far_fit_vectorial_lightcurve

    ulc = build_unified_lightcurve(
        scp=scp,
        stacking_method=stacking_method,
        vectorial_fitting_requires_km=swift_project_config.vectorial_fitting_requires_km,
        vectorial_fit_source=vectorial_fit_source,
        dust_redness_prior_mean=get_dust_redness_mean_prior(),
        dust_redness_prior_sigma=get_dust_redness_sigma_prior(),
    )
    if ulc is None:
        print("Unable to build unified lightcurve!")
        return

    # print(ulc)

    ulc_product = scp.get_product(
        PipelineFilesEnum.unified_lightcurve, stacking_method=stacking_method
    )
    assert ulc_product is not None
    ulc_product.data = lightcurve_to_dataframe(lc=ulc)
    ulc_product.write()
