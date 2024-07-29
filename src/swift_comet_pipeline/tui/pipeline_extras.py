from enum import StrEnum

import numpy as np

from swift_comet_pipeline.aperture.q_vs_aperture_kde_seaborn import (
    show_q_density_estimates_vs_redness,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius import (
    get_production_plateaus_from_yaml,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius_entry import (
    q_vs_aperture_radius_entry_list_from_dataframe,
)
from swift_comet_pipeline.lightcurve.lightcurve import dataframe_to_lightcurve
from swift_comet_pipeline.lightcurve.lightcurve_aperture import (
    lightcurve_from_aperture_plateaus,
)
from swift_comet_pipeline.lightcurve.lightcurve_matplotlib import show_lightcurve_mpl
from swift_comet_pipeline.orbits.perihelion import find_perihelion
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.tui.pipeline_extras_epoch_summary import (
    pipeline_extra_epoch_summary,
    pipeline_extra_latex_table_summary,
)
from swift_comet_pipeline.tui.pipeline_extras_orbital_data import (
    pipeline_extra_orbital_data,
)
from swift_comet_pipeline.tui.pipeline_extras_status import pipeline_extra_status
from swift_comet_pipeline.tui.tui_common import (
    clear_screen,
    get_selection,
    stacked_epoch_menu,
    wait_for_key,
)


class PipelineExtrasMenuEntry(StrEnum):
    pipeline_status = "pipeline status"
    epoch_summary = "epoch summary"
    epoch_latex_observation_log = "observation summary in latex format"
    get_orbital_data = "query jpl for comet and earth orbital data"
    comet_centers = "Raw images with comet centers marked"
    show_aperture_lightcurve = "show aperture lightcurve"
    show_vectorial_lightcurves = "show vectorial lightcurves"
    show_ridgeplot = "show ridgeplot"

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
        elif step == PipelineExtrasMenuEntry.get_orbital_data:
            pipeline_extra_orbital_data(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.comet_centers:
            mark_comet_centers(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_aperture_lightcurve:
            show_aperture_lightcurve(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_vectorial_lightcurves:
            show_vectorial_lightcurves(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.show_ridgeplot:
            show_ridgeplot(swift_project_config=swift_project_config)
        else:
            exit_menu = True

        wait_for_key()


def mark_comet_centers(swift_project_config: SwiftProjectConfig) -> None:
    pass
    # """
    # Finds images in uvv and uw1 filters, and outputs pngs images of each
    # observation annotated with the center of the comet marked.
    # Output images are placed in image_save_dir/[filter]/
    # """
    #
    # pipeline_files = PipelineFiles(
    #     base_product_save_path=swift_project_config.project_path
    # )
    #
    # swift_data = SwiftData(swift_project_config.swift_data_path)
    #
    # # select the epoch we want to process
    # epoch_id = epoch_menu(pipeline_files=pipeline_files)
    # if epoch_id is None:
    #     print("no epoch selected!")
    #     wait_for_key()
    #     return
    #
    # # load the epoch database
    # epoch = pipeline_files.read_pipeline_product(
    #     PipelineProductType.epoch, epoch_id=epoch_id
    # )
    # if epoch is None:
    #     print("Error loading epoch!")
    #     wait_for_key()
    #     return
    #
    # image_save_dir: pathlib.Path = swift_project_config.project_path / "centers"
    # plt.rcParams["figure.figsize"] = (15, 15)
    #
    # # directories to store the uw1 and uvv images: image_save_dir/[filter]/
    # dir_by_filter = {
    #     SwiftFilter.uw1: image_save_dir
    #     / pathlib.Path(filter_to_file_string(SwiftFilter.uw1)),
    #     SwiftFilter.uvv: image_save_dir
    #     / pathlib.Path(filter_to_file_string(SwiftFilter.uvv)),
    # }
    # # create directories we will need if they don't exist
    # for fdir in dir_by_filter.values():
    #     fdir.mkdir(parents=True, exist_ok=True)
    #
    # progress_bar = tqdm(epoch.iterrows(), total=epoch.shape[0])
    # for _, row in progress_bar:
    #     obsid = row["OBS_ID"]
    #     extension = row["EXTENSION"]
    #     px = round(float(row["PX"]))
    #     py = round(float(row["PY"]))
    #     filter_type = row.FILTER
    #     filter_str = filter_to_file_string(filter_type)  # type: ignore
    #
    #     # ask where the raw swift data FITS file is and read it
    #     image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore
    #     image_data = fits.getdata(image_path, ext=row["EXTENSION"])
    #
    #     output_image_name = pathlib.Path(f"{obsid}_{extension}_{filter_str}.png")
    #     output_image_path = dir_by_filter[filter_type] / output_image_name  # type: ignore
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 1, 1)
    #
    #     zscale = ZScaleInterval()
    #     vmin, vmax = zscale.get_limits(image_data)
    #
    #     im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)  # type: ignore
    #     fig.colorbar(im1)
    #     # mark comet center
    #     plt.axvline(px, color="w", alpha=0.3)
    #     plt.axhline(py, color="w", alpha=0.3)
    #
    #     ax1.set_title("C/2013US10")
    #     ax1.set_xlabel(f"{row['MID_TIME']}")
    #     ax1.set_ylabel(f"{row['FITS_FILENAME']}")
    #
    #     plt.savefig(output_image_path)
    #     plt.close()
    #     # num_processed += 1
    #
    #     progress_bar.set_description(f"Processed {obsid} extension {extension}")
    #
    # print("")


def show_aperture_lightcurve(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    stacking_method = StackingMethod.summation

    t_perihelion_list = find_perihelion(data_ingestion_files=data_ingestion_files)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return
    t_perihelion = t_perihelion_list[0].t_perihelion

    # aperture productions
    aperture_lc = lightcurve_from_aperture_plateaus(
        pipeline_files=pipeline_files,
        stacking_method=stacking_method,
        t_perihelion=t_perihelion,
    )

    if aperture_lc is None:
        return

    # best near-fit lightcurve
    pipeline_files.best_near_fit_lightcurves[
        stacking_method
    ].read_product_if_not_loaded()
    vectorial_near_fit_df = pipeline_files.best_near_fit_lightcurves[
        stacking_method
    ].data

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
    # near_fit_df.rename(
    #     columns={"near_fit_q": "q", "near_fit_q_err": "q_err"}, inplace=True
    # )
    # vectorial_lcs = dataframe_to_lightcurve(df=near_fit_df)

    # lc_total = aperture_lc + vectorial_lcs
    lc_total = aperture_lc

    show_lightcurve_mpl(lc=lc_total, best_lc=vectorial_near_fit_lc)
    # show_lightcurve_mpl(lc=lc_total, best_lc=vectorial_near_fit_lc + vectorial_far_fit_lc)
    # show_lightcurve_mpl(lc=lc_total)


def show_vectorial_lightcurves(swift_project_config: SwiftProjectConfig) -> None:

    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    # TODO: stacking method selection
    stacking_method = StackingMethod.summation

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    # TODO: selection menu for which fits to see

    # all vectorial fits
    pipeline_files.complete_vectorial_lightcurves[
        stacking_method
    ].read_product_if_not_loaded()
    complete_vectorial_df = pipeline_files.complete_vectorial_lightcurves[
        stacking_method
    ].data

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
    near_fit_df.rename(
        columns={"near_fit_q": "q", "near_fit_q_err": "q_err"}, inplace=True
    )
    vectorial_near_lcs = dataframe_to_lightcurve(df=near_fit_df)

    # best near-fit lightcurve
    pipeline_files.best_near_fit_lightcurves[
        stacking_method
    ].read_product_if_not_loaded()
    vectorial_best_near_fit_df = pipeline_files.best_near_fit_lightcurves[
        stacking_method
    ].data

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
    far_fit_df.rename(
        columns={"far_fit_q": "q", "far_fit_q_err": "q_err"}, inplace=True
    )
    vectorial_far_lcs = dataframe_to_lightcurve(df=far_fit_df)

    # best far-fit lightcurve
    pipeline_files.best_far_fit_lightcurves[
        stacking_method
    ].read_product_if_not_loaded()
    vectorial_best_far_fit_df = pipeline_files.best_far_fit_lightcurves[
        stacking_method
    ].data

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
    # full_fit_df.rename(
    #     columns={"full_fit_q": "q", "full_fit_q_err": "q_err"}, inplace=True
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


def show_ridgeplot(swift_project_config: SwiftProjectConfig) -> None:

    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    parent_epoch = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    )
    if parent_epoch is None:
        return

    epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        parent_epoch=parent_epoch
    )
    if epoch_subpipeline is None:
        return

    epoch_subpipeline.stacked_epoch.read()
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        print("Error reading epoch!")
        return

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    stacking_method = StackingMethod.summation

    epoch_subpipeline.qh2o_vs_aperture_radius_analyses[
        stacking_method
    ].read_product_if_not_loaded()
    df = epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].data

    q_vs_r = q_vs_aperture_radius_entry_list_from_dataframe(df=df)
    q_plateau_list_dict = get_production_plateaus_from_yaml(yaml_dict=df.attrs)

    # by_dust_redness = lambda x: x.dust_redness
    # sorted_q_vs_r = sorted(q_vs_r, key=by_dust_redness)  # type: ignore

    show_q_density_estimates_vs_redness(
        q_vs_aperture_radius_list=q_vs_r,
        q_plateau_list_dict=q_plateau_list_dict,
        km_per_pix=km_per_pix,
    )

    # show_q_vs_aperture_ridgeplot(
    #     sorted_q_vs_aperture_radius_list=sorted_q_vs_r,
    #     q_plateau_list_dict=q_plateau_list_dict,
    #     km_per_pix=km_per_pix,
    # )
