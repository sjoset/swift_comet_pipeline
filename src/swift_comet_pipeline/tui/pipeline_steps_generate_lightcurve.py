# import itertools

import numpy as np
import pandas as pd
import astropy.units as u
from tqdm import tqdm

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import lightcurve_to_dataframe
from swift_comet_pipeline.lightcurve.lightcurve_vectorial import (
    lightcurve_from_vectorial_fits,
)
from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


def find_minimum_percent_error_in_lightcurve(
    df: pd.DataFrame, production_column_name: str, production_err_column_name: str
) -> pd.DataFrame:

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


def generate_lightcurve_step(swift_project_config: SwiftProjectConfig) -> None:

    pd.options.display.float_format = "{:3.2e}".format
    pipeline_files = PipelineFiles(swift_project_config.project_path)

    # stacking_methods = [StackingMethod.summation, StackingMethod.median]
    # selection = get_selection(stacking_methods)
    # if selection is None:
    #     return
    # stacking_method = stacking_methods[selection]

    stacking_method = StackingMethod.summation

    data_ingestion_files = pipeline_files.data_ingestion_files

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    t_perihelion_list = find_perihelion(data_ingestion_files=data_ingestion_files)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    # TODO: magic number
    near_far_radius = 50000 * u.km  # type: ignore

    # TODO: magic numbers
    dust_rednesses = [
        DustReddeningPercent(x) for x in np.linspace(0.0, 50.0, num=51, endpoint=True)
    ]

    near_fit_lcs = {}
    far_fit_lcs = {}
    full_fit_lcs = {}
    dust_progress_bar = tqdm(dust_rednesses)
    for dust_redness in dust_progress_bar:
        dust_progress_bar.set_description(f"Dust redness: {dust_redness}")
        near_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            pipeline_files=pipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.near_fit,
            near_far_radius=near_far_radius,
        )
        far_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            pipeline_files=pipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.far_fit,
            near_far_radius=near_far_radius,
        )
        full_fit_lcs[dust_redness] = lightcurve_from_vectorial_fits(
            pipeline_files=pipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.full_fit,
            near_far_radius=near_far_radius,
        )

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

    for lc_df_data, lc_product in zip(
        [
            best_near_fit_lightcurve_df,
            best_far_fit_lightcurve_df,
            best_full_fit_lightcurve_df,
        ],
        [
            pipeline_files.best_near_fit_lightcurves[stacking_method],
            pipeline_files.best_far_fit_lightcurves[stacking_method],
            pipeline_files.best_full_fit_lightcurves[stacking_method],
        ],
    ):
        lc_product.data = lc_df_data
        lc_product.write()

    # rename conflicting columns so that we can combine them all into one
    df_near.rename(columns={"q": "near_fit_q", "q_err": "near_fit_q_err"}, inplace=True)
    df_far.rename(columns={"q": "far_fit_q", "q_err": "far_fit_q_err"}, inplace=True)
    df_full.rename(columns={"q": "full_fit_q", "q_err": "full_fit_q_err"}, inplace=True)

    # drop the duplicated columns: each df has its own time_from_perihelion and observation_time columns
    complete_vectorial_lightcurves_df = pd.concat([df_near, df_far, df_full], axis=1)
    complete_vectorial_lightcurves_df = complete_vectorial_lightcurves_df.loc[
        :, ~complete_vectorial_lightcurves_df.columns.duplicated()
    ]

    pipeline_files.complete_vectorial_lightcurves[stacking_method].data = (
        complete_vectorial_lightcurves_df
    )
    pipeline_files.complete_vectorial_lightcurves[stacking_method].write()


# def show_lightcurves(df) -> None:
#
#     dust_rednesses = [DustReddeningPercent(x) for x in set(df.assumed_redness_percent)]
#
#     dust_cmap = LinearSegmentedColormap.from_list(
#         name="custom", colors=["#8e8e8e", "#bb0000"], N=(len(dust_rednesses) + 1)
#     )
#     dust_line_colors = dust_cmap(
#         np.array(dust_rednesses).astype(np.float32) / DustReddeningPercent(100.0)
#     )
#
#     fig, axs = plt.subplots(2, 4)
#
#     print(axs)
#     print(list(axs))
#     print(np.ravel(axs))
#     for dust_redness, line_color, ax in zip(
#         dust_rednesses, dust_line_colors, np.ravel(axs)
#     ):
#         ax.plot(
#             df.time_from_perihelion,
#             df.far_fit_q,
#             label=f"Q(H20) at {dust_redness=}",
#             color=line_color,
#             alpha=0.65,
#         )
#
#     # ax.set_xscale("log")
#     # ax.set_yscale("log")
#     ax.set_xlabel("days from perihelion")
#     ax.set_ylabel("Q(H2O)")
#     ax.legend()
#     # fig.suptitle(
#     #     f"Rh: {helio_r.to_value(u.AU):1.4f} AU, Delta: {delta.to_value(u.AU):1.4f} AU,\nTime from perihelion: {time_from_perihelion.to_value(u.day)} days\nfitting data from {fit_begin_r.to(u.km):1.3e} to {fit_end_r.to(u.km):1.3e}"  # type: ignore
#     # )
#     plt.show()
#     plt.close()
