from enum import StrEnum

import numpy as np

from swift_comet_pipeline.aperture.plateau_distribution_seaborn import (
    show_plateau_distribution_seaborn,
)
from swift_comet_pipeline.aperture.q_vs_aperture_kde_seaborn import (
    show_q_density_estimates_vs_redness,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius import (
    get_production_plateaus_from_yaml,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius_entry import (
    q_vs_aperture_radius_entry_list_from_dataframe,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius_seaborn import (
    show_q_vs_aperture_radius_seaborn,
)
from swift_comet_pipeline.lightcurve.lightcurve import dataframe_to_lightcurve
from swift_comet_pipeline.lightcurve.lightcurve_aperture import (
    lightcurve_from_aperture_plateaus,
)
from swift_comet_pipeline.lightcurve.lightcurve_matplotlib import show_lightcurve_mpl
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.tui.pipeline_extras_epoch_summary import (
    pipeline_extra_epoch_summary,
    pipeline_extra_latex_table_summary,
)
from swift_comet_pipeline.tui.pipeline_extras_status import pipeline_extra_status
from swift_comet_pipeline.tui.tui_common import (
    clear_screen,
    get_selection,
    wait_for_key,
)
from swift_comet_pipeline.tui.tui_menus import stacked_epoch_menu


class PipelineExtrasMenuEntry(StrEnum):
    pipeline_status = "pipeline status"
    epoch_summary = "epoch summary"
    epoch_latex_observation_log = "observation summary in latex format"
    show_aperture_lightcurve = "show aperture lightcurve"
    show_vectorial_lightcurves = "show vectorial lightcurves"
    show_q_histograms_vs_redness = "show q(H2O) histograms vs. dust redness"
    show_q_vs_aperture_radius = "show q vs aperture radius as function of redness"
    show_plateau_distribution = (
        "show distribution of production plateaus as a function of redness"
    )

    @classmethod
    def all_extras(cls):
        return [x for x in cls]


def pipeline_extras_menu(swift_project_config: SwiftProjectConfig) -> None:
    exit_menu = False
    extras_menu_entries = PipelineExtrasMenuEntry.all_extras()
    while not exit_menu:
        clear_screen()
        step_selection = get_selection(extras_menu_entries)
        if step_selection is None:
            exit_menu = True
            continue
        step = extras_menu_entries[step_selection]

        if step == PipelineExtrasMenuEntry.pipeline_status:
            pipeline_extra_status(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.epoch_summary:
            pipeline_extra_epoch_summary(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.epoch_latex_observation_log:
            pipeline_extra_latex_table_summary(
                swift_project_config=swift_project_config
            )
        elif step == PipelineExtrasMenuEntry.show_aperture_lightcurve:
            show_aperture_lightcurve(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_vectorial_lightcurves:
            show_vectorial_lightcurves(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_q_histograms_vs_redness:
            show_q_histograms_vs_redness(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_q_vs_aperture_radius:
            show_q_vs_aperture_radius(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_plateau_distribution:
            show_plateau_distribution(swift_project_config=swift_project_config)
        else:
            exit_menu = True

        wait_for_key()


def show_aperture_lightcurve(swift_project_config: SwiftProjectConfig) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # TODO: stacking method option
    stacking_method = StackingMethod.summation

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    # aperture productions
    aperture_lc = lightcurve_from_aperture_plateaus(
        scp=scp,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
    )

    if aperture_lc is None:
        return

    # best near-fit lightcurve
    vectorial_near_fit_df = scp.get_product_data(
        pf=PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
        stacking_method=stacking_method,
        # fit_type=VectorialFitType.near_fit,
    )
    if vectorial_near_fit_df is None:
        return
    vectorial_near_fit_lc = dataframe_to_lightcurve(df=vectorial_near_fit_df)

    # # best far-fit lightcurve
    # pipeline_files.best_far_fit_lightcurve.read_product_if_not_loaded()
    # vectorial_far_fit_df = pipeline_files.best_far_fit_lightcurve.data
    #
    # if vectorial_far_fit_df is None:
    #     return
    #
    # vectorial_far_fit_lc = dataframe_to_lightcurve(df=vectorial_far_fit_df)

    # # all vectorial fits
    # pipeline_files.complete_vectorial_lightcurve.read_product_if_not_loaded()
    # complete_vectorial_df = pipeline_files.complete_vectorial_lightcurve.data
    #
    # if complete_vectorial_df is None:
    #     return

    # # pull out all near fits
    # near_fit_df = complete_vectorial_df[
    #     [
    #         "observation_time",
    #         "time_from_perihelion_days",
    #         "rh_au",
    #         "near_fit_q",
    #         "near_fit_q_err",
    #         "dust_redness",
    #     ]
    # ].copy()
    # near_fit_df = near_fit_df.rename(
    #     columns={"near_fit_q": "q", "near_fit_q_err": "q_err"}
    # )
    # vectorial_lcs = dataframe_to_lightcurve(df=near_fit_df)

    # lc_total = aperture_lc + vectorial_lcs
    lc_total = aperture_lc

    show_lightcurve_mpl(lc=lc_total, best_lc=vectorial_near_fit_lc)
    # show_lightcurve_mpl(lc=lc_total, best_lc=vectorial_near_fit_lc + vectorial_far_fit_lc)
    # show_lightcurve_mpl(lc=lc_total)


def show_vectorial_lightcurves(swift_project_config: SwiftProjectConfig) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # pipeline_files = PipelineFiles(swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    # TODO: stacking method selection
    stacking_method = StackingMethod.summation

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return
    #
    # if epoch_subpipeline_files is None:
    #     print("No epochs available to stack!")
    #     return

    # TODO: selection menu for which fits to see

    # # all vectorial fits
    # pipeline_files.complete_vectorial_lightcurves[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # complete_vectorial_df = pipeline_files.complete_vectorial_lightcurves[
    #     stacking_method
    # ].data

    complete_vectorial_df = scp.get_product_data(
        pf=PipelineFilesEnum.complete_vectorial_lightcurve,
        stacking_method=stacking_method,
    )

    if complete_vectorial_df is None:
        return

    # pull out all near fits
    near_fit_df = complete_vectorial_df[
        [
            "observation_time",
            "time_from_perihelion_days",
            "rh_au",
            "near_fit_q",
            "near_fit_q_err",
            "dust_redness",
        ]
    ].copy()
    near_fit_df = near_fit_df.rename(
        columns={"near_fit_q": "q", "near_fit_q_err": "q_err"}
    )
    vectorial_near_lcs = dataframe_to_lightcurve(df=near_fit_df)

    # # best near-fit lightcurve
    # pipeline_files.best_near_fit_lightcurves[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # vectorial_best_near_fit_df = pipeline_files.best_near_fit_lightcurves[
    #     stacking_method
    # ].data

    vectorial_best_near_fit_df = scp.get_product_data(
        pf=PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
        stacking_method=stacking_method,
        # fit_type=VectorialFitType.near_fit,
    )
    if vectorial_best_near_fit_df is None:
        return
    vectorial_best_near_fit_lc = dataframe_to_lightcurve(df=vectorial_best_near_fit_df)

    # pull out all far fits
    far_fit_df = complete_vectorial_df[
        [
            "observation_time",
            "time_from_perihelion_days",
            "rh_au",
            "far_fit_q",
            "far_fit_q_err",
            "dust_redness",
        ]
    ].copy()
    far_fit_df = far_fit_df.rename(columns={"far_fit_q": "q", "far_fit_q_err": "q_err"})
    vectorial_far_lcs = dataframe_to_lightcurve(df=far_fit_df)

    # # best far-fit lightcurve
    # pipeline_files.best_far_fit_lightcurves[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # vectorial_best_far_fit_df = pipeline_files.best_far_fit_lightcurves[
    #     stacking_method
    # ].data

    vectorial_best_far_fit_df = scp.get_product_data(
        pf=PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
        stacking_method=stacking_method,
        # fit_type=VectorialFitType.far_fit,
    )
    if vectorial_best_far_fit_df is None:
        return
    vectorial_best_far_fit_lc = dataframe_to_lightcurve(df=vectorial_best_far_fit_df)

    # # pull out all full fits
    # full_fit_df = complete_vectorial_df[
    #     [
    #         "observation_time",
    #         "time_from_perihelion_days",
    #         "rh_au",
    #         "full_fit_q",
    #         "full_fit_q_err",
    #         "dust_redness",
    #     ]
    # ].copy()
    # full_fit_df = full_fit_df.rename(
    #     columns={"full_fit_q": "q", "full_fit_q_err": "q_err"}
    # )
    # vectorial_lcs = dataframe_to_lightcurve(df=full_fit_df)
    #
    # # best full-fit lightcurve
    # pipeline_files.best_full_fit_lightcurve.read_product_if_not_loaded()
    # vectorial_full_fit_df = pipeline_files.best_full_fit_lightcurve.data
    #
    # if vectorial_full_fit_df is None:
    #     return
    #
    # vectorial_full_fit_lc = dataframe_to_lightcurve(df=vectorial_full_fit_df)

    show_lightcurve_mpl(
        lc=vectorial_near_lcs + vectorial_far_lcs,
        best_lc=vectorial_best_near_fit_lc + vectorial_best_far_fit_lc,
    )


def show_q_histograms_vs_redness(swift_project_config: SwiftProjectConfig) -> None:

    # pipeline_files = PipelineFiles(swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return
    #
    # if epoch_subpipeline_files is None:
    #     print("No epochs available to stack!")
    #     return

    # parent_epoch = stacked_epoch_menu(
    #     pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    # )
    # if parent_epoch is None:
    #     return

    epoch_id_selected = stacked_epoch_menu(scp=scp)

    # epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #     parent_epoch=parent_epoch
    # )
    # if epoch_subpipeline is None:
    #     return

    # epoch_subpipeline.stacked_epoch.read()
    # stacked_epoch = epoch_subpipeline.stacked_epoch.data
    # if stacked_epoch is None:
    #     print("Error reading epoch!")
    #     return

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    # TODO: stacking method
    stacking_method = StackingMethod.summation

    # epoch_subpipeline.qh2o_vs_aperture_radius_analyses[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # df = epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].data

    df = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id_selected,
        stacking_method=stacking_method,
    )
    assert df is not None

    q_vs_r = q_vs_aperture_radius_entry_list_from_dataframe(df=df)
    q_plateau_list_dict = get_production_plateaus_from_yaml(yaml_dict=df.attrs)

    show_q_density_estimates_vs_redness(
        q_vs_aperture_radius_list=q_vs_r,
        q_plateau_list_dict=q_plateau_list_dict,
    )


def show_q_vs_aperture_radius(swift_project_config: SwiftProjectConfig) -> None:

    # pipeline_files = PipelineFiles(swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return
    #
    # if epoch_subpipeline_files is None:
    #     print("No epochs available to stack!")
    #     return

    epoch_id_selected = stacked_epoch_menu(scp=scp)
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    # parent_epoch = stacked_epoch_menu(
    #     pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    # )
    # if parent_epoch is None:
    #     return
    #
    # epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #     parent_epoch=parent_epoch
    # )
    # if epoch_subpipeline is None:
    #     return

    # epoch_subpipeline.stacked_epoch.read()
    # stacked_epoch = epoch_subpipeline.stacked_epoch.data
    # if stacked_epoch is None:
    #     print("Error reading epoch!")
    #     return

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    stacking_method = StackingMethod.summation

    # epoch_subpipeline.qh2o_vs_aperture_radius_analyses[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # df = epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].data

    df = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id_selected,
        stacking_method=stacking_method,
    )
    assert df is not None

    q_vs_r = q_vs_aperture_radius_entry_list_from_dataframe(df=df)
    q_plateau_list_dict = get_production_plateaus_from_yaml(yaml_dict=df.attrs)

    show_q_vs_aperture_radius_seaborn(
        q_vs_aperture_radius_list=q_vs_r,
        q_plateau_list_dict=q_plateau_list_dict,
        km_per_pix=km_per_pix,
    )


def show_plateau_distribution(swift_project_config: SwiftProjectConfig) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # pipeline_files = PipelineFiles(swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipeline_files = pipeline_files.epoch_subpipelines
    #
    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return
    #
    # if epoch_subpipeline_files is None:
    #     print("No epochs available to stack!")
    #     return

    epoch_id_selected = stacked_epoch_menu(scp=scp)
    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    # parent_epoch = stacked_epoch_menu(
    #     pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    # )
    # if parent_epoch is None:
    #     return
    #
    # epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #     parent_epoch=parent_epoch
    # )
    # if epoch_subpipeline is None:
    #     return
    #
    # epoch_subpipeline.stacked_epoch.read()
    # stacked_epoch = epoch_subpipeline.stacked_epoch.data
    # if stacked_epoch is None:
    #     print("Error reading epoch!")
    #     return

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    stacking_method = StackingMethod.summation

    df = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id_selected,
        stacking_method=stacking_method,
    )
    assert df is not None

    # epoch_subpipeline.qh2o_vs_aperture_radius_analyses[
    #     stacking_method
    # ].read_product_if_not_loaded()
    # df = epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].data

    q_vs_r = q_vs_aperture_radius_entry_list_from_dataframe(df=df)
    q_plateau_list_dict = get_production_plateaus_from_yaml(yaml_dict=df.attrs)

    show_plateau_distribution_seaborn(
        q_vs_aperture_radius_list=q_vs_r,
        q_plateau_list_dict=q_plateau_list_dict,
        km_per_pix=km_per_pix,
    )
