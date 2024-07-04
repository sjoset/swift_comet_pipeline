from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from itertools import chain
import pathlib

import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)
from swift_comet_pipeline.comet.comet_radial_profile import (
    radial_profile_from_dataframe_product,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import LightCurve, LightCurveEntry
from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


# # TODO: rename to lightcurve and just have three separate lists for near, far, and full
# @dataclass
# class EpochProductionReport:
#     observation_time: Time
#     time_from_perihelion: u.Quantity
#
#     near_fit_q: float
#     near_fit_q_err: float
#
#     far_fit_q: float
#     far_fit_q_err: float
#
#     full_fit_q: float
#     full_fit_q_err: float
#
#     assumed_redness_percent: float


class VectorialFitType(StrEnum):
    near_fit = auto()
    far_fit = auto()
    full_fit = auto()

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]


def read_product_if_not_loaded(p: PipelineProduct) -> None:

    if p.product_path.exists() is False:
        # print(f"{p.product_path} does not exist!")
        return

    if p.data is None:
        p.read()


def get_lightcurve(
    pipeline_files: PipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurve | None:

    if pipeline_files.epoch_subpipelines is None:
        return None

    lightcurve = []
    for epoch_subpipeline in pipeline_files.epoch_subpipelines:
        lc_entry = lightcurve_entry_from_epoch_subpipeline(
            epoch_subpipeline=epoch_subpipeline,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=fit_type,
            near_far_radius=near_far_radius,
        )
        lightcurve.append(lc_entry)

    return lightcurve


def lightcurve_entry_from_epoch_subpipeline(
    epoch_subpipeline: EpochSubpipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurveEntry | None:

    if not epoch_subpipeline.stacked_epoch.product_path.exists():
        # print("Could not find a stack for this epoch! Skipping.")
        return None

    read_product_if_not_loaded(epoch_subpipeline.stacked_epoch)
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        # print("Error reading epoch!")
        return None

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU
    observation_time = Time(np.mean(stacked_epoch.MID_TIME))

    read_product_if_not_loaded(
        epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method]
    )
    read_product_if_not_loaded(
        epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method]
    )

    uw1_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].data
    )
    uvv_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].data
    )

    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r, model_backend="rust")
    if vmr.column_density_interpolation is None:
        print(
            "No column density interpolation returned from vectorial model! This is a bug! Exiting."
        )
        exit(1)

    # TODO: magic number, empirically determined after some testing with a few datasets
    # near_far_radius = 50000 * u.km  # type: ignore

    ccd = calculate_comet_column_density(
        stacked_epoch=stacked_epoch,
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
        r_min=1 * u.km,  # type: ignore
    )

    if fit_type == VectorialFitType.near_fit:
        fit_radius_start = 1 * u.km  # type: ignore
        fit_radius_stop = near_far_radius  # type: ignore
    elif fit_type == VectorialFitType.far_fit:
        fit_radius_start = near_far_radius  # type: ignore
        # TODO: magic number - this should be large enough for all cases but there should be a better way to set upper limit
        fit_radius_stop = 1.0e10 * u.km  # type: ignore
    elif fit_type == VectorialFitType.full_fit:
        fit_radius_start = 1 * u.km  # type: ignore
        fit_radius_stop = 1.0e10 * u.km  # type: ignore

    vec_fit = vectorial_fit(
        comet_column_density=ccd,
        model_Q=model_Q,
        vmr=vmr,
        r_fit_min=fit_radius_start,
        r_fit_max=fit_radius_stop,
    )

    return LightCurveEntry(
        observation_time=observation_time,
        time_from_perihelion=observation_time - t_perihelion,
        q=vec_fit.best_fit_Q.to_value(1 / u.s),  # type: ignore
        q_err=vec_fit.best_fit_Q_err.to_value(1 / u.s),  # type: ignore
        dust_redness=dust_redness,
    )


# def lightcurve_from_epoch(
#     epoch_subpipeline: EpochSubpipelineFiles,
#     stacking_method: StackingMethod,
#     t_perihelion: Time,
#     dust_rednesses: list[DustReddeningPercent],
# ) -> LightCurve | None:
#
#     if not epoch_subpipeline.stacked_epoch.product_path.exists():
#         print("Could not find a stack for this epoch! Skipping.")
#         return None
#
#     # TODO: check if epoch is already read in
#     epoch_subpipeline.stacked_epoch.read()
#     stacked_epoch = epoch_subpipeline.stacked_epoch.data
#     if stacked_epoch is None:
#         print("Error reading epoch!")
#         return None
#
#     helio_r = np.mean(stacked_epoch.HELIO) * u.AU
#     observation_time = Time(np.mean(stacked_epoch.MID_TIME))
#
#     # TODO: check that these exist
#     epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].read()
#     epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].read()
#
#     uw1_profile = radial_profile_from_dataframe_product(
#         df=epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].data
#     )
#     uvv_profile = radial_profile_from_dataframe_product(
#         df=epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].data
#     )
#
#     # print("Running vectorial model...")
#     model_Q = 1e29 / u.s  # type: ignore
#     vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r, model_backend="rust")
#     if vmr.column_density_interpolation is None:
#         print(
#             "No column density interpolation returned from vectorial model! This is a bug! Exiting."
#         )
#         exit(1)
#
#     # TODO: magic number, empirically determined after some testing with a few datasets
#     near_far_radius = 50000 * u.km  # type: ignore
#
#     ccds = {}
#     near_fits = {}
#     far_fits = {}
#     full_fits = {}
#     for dust_redness in dust_rednesses:
#         ccds[dust_redness] = calculate_comet_column_density(
#             stacked_epoch=stacked_epoch,
#             uw1_profile=uw1_profile,
#             uvv_profile=uvv_profile,
#             dust_redness=dust_redness,
#             r_min=1 * u.km,  # type: ignore
#         )
#         near_fits[dust_redness] = vectorial_fit(
#             comet_column_density=ccds[dust_redness],
#             model_Q=model_Q,
#             vmr=vmr,
#             r_fit_min=1 * u.km,  # type: ignore
#             r_fit_max=near_far_radius,
#         )
#         far_fits[dust_redness] = vectorial_fit(
#             comet_column_density=ccds[dust_redness],
#             model_Q=model_Q,
#             vmr=vmr,
#             r_fit_min=near_far_radius,
#             r_fit_max=1.0e10 * u.km,  # type: ignore
#         )
#         full_fits[dust_redness] = vectorial_fit(
#             comet_column_density=ccds[dust_redness],
#             model_Q=model_Q,
#             vmr=vmr,
#             r_fit_min=1 * u.km,  # type: ignore
#             r_fit_max=1.0e10 * u.km,  # type: ignore
#         )
#
#     # print(f"Heliocentric distance: {helio_r.to_value(u.AU):1.4f} AU")  # type: ignore
#     # print(f"Date: {observation_time}")
#     # print(f"Perihelion: {t_perihelion}")
#     # print("Near-nucleus vectorial model fitting:")
#     # for dust_redness in dust_rednesses:
#     #     print(
#     #         f"Redness: {dust_redness}\tQ: {near_fits[dust_redness].best_fit_Q:1.3e}\tErr: {near_fits[dust_redness].best_fit_Q_err:1.3e}"
#     #     )
#     #
#     # print("Far from nucleus vectorial model fitting:")
#     # for dust_redness in dust_rednesses:
#     #     print(
#     #         f"Redness: {dust_redness}\tQ: {far_fits[dust_redness].best_fit_Q:1.3e}\tErr: {far_fits[dust_redness].best_fit_Q_err:1.3e}"
#     #     )
#     #
#     # print("Whole curve vectorial model fitting:")
#     # for dust_redness in dust_rednesses:
#     #     print(
#     #         f"Redness: {dust_redness}\tQ: {full_fits[dust_redness].best_fit_Q:1.3e}\tErr: {full_fits[dust_redness].best_fit_Q_err:1.3e}"
#     #     )
#
#     reports = [
#         EpochProductionReport(
#             observation_time=observation_time,
#             time_from_perihelion=observation_time - t_perihelion,
#             near_fit_q=near_fits[dr].best_fit_Q.to_value(1 / u.s),
#             near_fit_q_err=near_fits[dr].best_fit_Q_err.to_value(1 / u.s),
#             far_fit_q=far_fits[dr].best_fit_Q.to_value(1 / u.s),
#             far_fit_q_err=far_fits[dr].best_fit_Q_err.to_value(1 / u.s),
#             full_fit_q=full_fits[dr].best_fit_Q.to_value(1 / u.s),
#             full_fit_q_err=full_fits[dr].best_fit_Q_err.to_value(1 / u.s),
#             assumed_redness_percent=dr,
#         )
#         for dr in dust_rednesses
#     ]
#
#     return reports


def lightcurve_to_dataframe(lc: LightCurve) -> pd.DataFrame:

    print(f"{lc=}")
    data_dict = [asdict(lc_entry) for lc_entry in lc if lc_entry is not None]
    df = pd.DataFrame(data=data_dict)
    return df


def generate_lightcurve_step(swift_project_config: SwiftProjectConfig) -> None:

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

    near_far_radius = 50000 * u.km  # type: ignore

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(-10.0, 40.0, num=51, endpoint=True)
        # DustReddeningPercent(x)
        # for x in np.linspace(-10.0, 40.0, num=6, endpoint=True)
    ]

    near_fit_lcs = {}
    far_fit_lcs = {}
    full_fit_lcs = {}
    for dust_redness in dust_rednesses:
        print(f"Calculating lightcurves for dust redness {dust_redness}...")
        near_fit_lcs[dust_redness] = get_lightcurve(
            pipeline_files=pipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.near_fit,
            near_far_radius=near_far_radius,
        )
        far_fit_lcs[dust_redness] = get_lightcurve(
            pipeline_files=pipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=VectorialFitType.far_fit,
            near_far_radius=near_far_radius,
        )
        full_fit_lcs[dust_redness] = get_lightcurve(
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

    df_near.rename(columns={"q": "near_fit_q", "q_err": "near_fit_q_err"}, inplace=True)
    df_far.rename(columns={"q": "far_fit_q", "q_err": "far_fit_q_err"}, inplace=True)
    df_full.rename(columns={"q": "full_fit_q", "q_err": "full_fit_q_err"}, inplace=True)

    # drop the repeating columns of observation_time and time_from_perihelion from the resulting dataframe,
    # by flipping them to rows, dropping dupes, and flipping back
    all_lightcurves_df = (
        pd.concat(
            [df_near, df_far, df_full],
            axis=1,
        )
        .T.drop_duplicates()
        .T
    )
    print(all_lightcurves_df)
    all_lightcurves_df.reset_index()

    lightcurve_output_path = pathlib.Path(f"lightcurves_{stacking_method}.csv")
    all_lightcurves_df.to_csv(lightcurve_output_path, index=False)

    # # dust_rednesses = [
    # #     DustReddeningPercent(x) for x in [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    # # ]
    #
    # # Pick which production source to use: near, far, or whole curve fitting
    # # q vs aperture also -- we need a way to pick out a production from that, and
    # # we need q vs aperture (at varying redness) to include the distant comets
    #
    # # parent_epoch = data_ingestion_files.epochs[0]
    #
    # all_reports = []
    # for parent_epoch in data_ingestion_files.epochs[:2]:
    #     print(f"Epoch: {parent_epoch.product_path.name}")
    #
    #     epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #         parent_epoch=parent_epoch
    #     )
    #     if epoch_subpipeline is None:
    #         return
    #
    #     epoch_production_report_list = lightcurve_from_epoch(
    #         epoch_subpipeline=epoch_subpipeline,
    #         stacking_method=stacking_method,
    #         t_perihelion=t_perihelion,
    #         dust_rednesses=dust_rednesses,
    #     )
    #
    #     if epoch_production_report_list is None:
    #         print(
    #             f"Could not generate report for epoch {parent_epoch.product_path.name}"
    #         )
    #         continue
    #
    #     all_reports.append(epoch_production_report_list)
    #
    # # flatten reports into one list
    # all_reports = list(chain.from_iterable(all_reports))
    #
    # # convert dataclasses to a dictionary to dump into a dataframe
    # report_dict = [asdict(x) for x in all_reports]
    # df = pd.DataFrame(data=report_dict)
    #
    # # by_red = df.groupby("assumed_redness_percent")
    # # for redness, redness_df in by_red:
    # #     print(f"Lightcurve at redness {redness}%:")
    # #     print(redness_df.loc[:, redness_df.columns != "assumed_redness_percent"])
    # #     print("")
    #
    # # TODO: loop through each epoch and judge redness at near, far, and full fits -> 3 'best' lightcurves
    #
    # # best near-fit lightcurve
    # best_near_fit_lightcurve = []
    # by_epoch = df.groupby("time_from_perihelion")
    # for tp, epoch_df in by_epoch:
    #     # print(epoch_df.near_fit_q_err / epoch_df.near_fit_q)
    #     idx_min = np.argmin(epoch_df.near_fit_q_err / epoch_df.near_fit_q)
    #
    #     percent_error = (
    #         epoch_df.iloc[idx_min].near_fit_q_err / epoch_df.iloc[idx_min].near_fit_q
    #     )
    #     print(f"{idx_min=}\t{percent_error=}")
    #     best_near_fit_lightcurve.append(epoch_df.iloc[idx_min])
    #
    # print("Best-fit near lightcurve")
    # # for e in best_near_fit_lightcurve:
    # #     print(e)
    # #     print("")
    # best_near_fit_lightcurve_df = pd.concat(best_near_fit_lightcurve)
    # print(best_near_fit_lightcurve_df.time_from_perihelion)
    # print(f"{best_near_fit_lightcurve_df=}")
    #
    # # df.to_csv("lightcurve.csv", index=False)


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
