import time
from itertools import product

import numpy as np
import pandas as pd
import astropy.units as u
from tqdm import tqdm

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import (
    LightCurve,
    dataframe_to_lightcurve,
    lightcurve_to_dataframe,
)
from swift_comet_pipeline.lightcurve.lightcurve_aperture import (
    lightcurve_from_aperture_plateaus,
)
from swift_comet_pipeline.lightcurve.lightcurve_bayesian import (
    bayesian_lightcurve_from_aperture_lightcurve,
    bayesian_lightcurve_to_dataframe,
)
from swift_comet_pipeline.lightcurve.lightcurve_vectorial import (
    lightcurve_from_vectorial_fits,
)
from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.tui.tui_common import get_selection


def find_minimum_percent_error_in_lightcurve(
    df: pd.DataFrame, production_column_name: str, production_err_column_name: str
) -> pd.DataFrame:
    # TODO: document this function and its inputs

    positive_production_mask = df.q > 0.0

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
    dust_mean_reddenings = np.linspace(0.0, 60.0, num=13, endpoint=True)
    dust_sigma_reddenings = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]

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


def parallel_vectorial_lightcurve_computer(
    dust_redness: DustReddeningPercent,
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    t_perihelion,
    fit_type: VectorialFitType,
    near_far_radius,
) -> tuple[DustReddeningPercent, LightCurve | None]:

    return dust_redness, lightcurve_from_vectorial_fits(
        scp=scp,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
        dust_redness=dust_redness,
        fit_type=fit_type,
        near_far_radius=near_far_radius,
    )


def build_vectorial_lightcurves_step(
    swift_project_config: SwiftProjectConfig, stacking_method: StackingMethod
) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # TODO: check for existence of files before computing

    # pd.options.display.float_format = "{:3.2e}".format

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    # TODO: magic number: make this an option in project config, with 50000 as a default
    near_far_radius = 50000 * u.km  # type: ignore

    # TODO: magic numbers
    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(-100.0, 100.0, num=201, endpoint=True)
    ]

    print("Precomputing vectorial models...")
    _ = lightcurve_from_vectorial_fits(
        scp=scp,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
        dust_redness=dust_rednesses[0],
        fit_type=VectorialFitType.near_fit,
        near_far_radius=near_far_radius,
    )

    print("Starting lightcurve calculations...")
    all_fit_start = time.perf_counter()
    near_fit_lcs = {}
    far_fit_lcs = {}
    full_fit_lcs = {}
    dust_progress_bar = tqdm(dust_rednesses)
    for dust_redness in dust_progress_bar:
        dust_progress_bar.set_description(f"Dust redness: {dust_redness}")
        near_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            scp=scp,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.near_fit,
            near_far_radius=near_far_radius,
        )
        far_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            scp=scp,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.far_fit,
            near_far_radius=near_far_radius,
        )
        full_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            scp=scp,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.full_fit,
            near_far_radius=near_far_radius,
        )
    all_fit_end = time.perf_counter()
    print(f"All fits done in {all_fit_end - all_fit_start:.2f} seconds")

    df_near = pd.concat(
        [lightcurve_to_dataframe(lc=near_fit_lcs[x]) for x in dust_rednesses]
    )
    df_far = pd.concat(
        [lightcurve_to_dataframe(lc=far_fit_lcs[x]) for x in dust_rednesses]
    )
    df_full = pd.concat(
        [lightcurve_to_dataframe(lc=full_fit_lcs[x]) for x in dust_rednesses]
    )

    best_near_fit_lightcurve_df = find_minimum_percent_error_in_lightcurve(
        df=df_near,
        production_column_name="q",
        production_err_column_name="q_err",
    )
    best_far_fit_lightcurve_df = find_minimum_percent_error_in_lightcurve(
        df=df_far,
        production_column_name="q",
        production_err_column_name="q_err",
    )
    best_full_fit_lightcurve_df = find_minimum_percent_error_in_lightcurve(
        df=df_full,
        production_column_name="q",
        production_err_column_name="q_err",
    )

    print("Lightcurve for near-nucleus fitting with best redness:")
    print(best_near_fit_lightcurve_df)
    print("Lightcurve for far-nucleus fitting with best redness:")
    print(best_far_fit_lightcurve_df)
    print("Lightcurve for full-nucleus fitting with best redness:")
    print(best_full_fit_lightcurve_df)

    for lc_df_data, pf in zip(
        [
            best_near_fit_lightcurve_df,
            best_far_fit_lightcurve_df,
            best_full_fit_lightcurve_df,
        ],
        [
            PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
            PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
            PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
        ],
    ):
        p = scp.get_product(pf=pf, stacking_method=stacking_method)
        assert p is not None
        p.data = lc_df_data
        p.write()

    # rename conflicting columns so that we can combine them all into one dataframe
    df_near = df_near.rename(columns={"q": "near_fit_q", "q_err": "near_fit_q_err"})
    df_far = df_far.rename(columns={"q": "far_fit_q", "q_err": "far_fit_q_err"})
    df_full = df_full.rename(columns={"q": "full_fit_q", "q_err": "full_fit_q_err"})

    # drop the duplicated columns: each df has its own time_from_perihelion and observation_time columns
    complete_vectorial_lightcurves_df = pd.concat([df_near, df_far, df_full], axis=1)
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
