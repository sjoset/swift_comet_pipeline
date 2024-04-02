from enum import StrEnum
import pathlib
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
from swift_comet_pipeline.comet_profile import (
    qh2o_from_surface_brightness_profiles,
    surface_brightness_profiles,
)

from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.flux_OH import beta_parameter
from swift_comet_pipeline.pipeline_extras_epoch_summary import (
    pipeline_extra_epoch_summary,
    pipeline_extra_latex_table_summary,
)
from swift_comet_pipeline.pipeline_extras_orbital_data import (
    pipeline_extra_orbital_data,
)
from swift_comet_pipeline.pipeline_extras_status import pipeline_extra_status
from swift_comet_pipeline.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
)
from swift_comet_pipeline.tui import (
    clear_screen,
    epoch_menu,
    get_selection,
    stacked_epoch_menu,
    wait_for_key,
)
from swift_comet_pipeline.pipeline_files import PipelineFiles, PipelineProductType


class PipelineExtrasMenuEntry(StrEnum):
    pipeline_status = "pipeline status"
    epoch_summary = "epoch summary"
    epoch_latex_observation_log = "observation summary in latex format"
    get_orbital_data = "query jpl for comet and earth orbital data"

    surf_brightness_test = "surface brightness test code"
    comet_centers = "Raw images with comet centers marked"

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
        elif step == PipelineExtrasMenuEntry.surf_brightness_test:
            do_surf_brightness_test(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.get_orbital_data:
            pipeline_extra_orbital_data(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.comet_centers:
            mark_comet_centers(swift_project_config=swift_project_config)
        else:
            exit_menu = True


def do_surf_brightness_test(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(
        base_product_save_path=swift_project_config.product_save_path
    )

    # select the epoch we want to process
    epoch_id = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_be=True
    )
    if epoch_id is None:
        return

    # load the epoch database
    epoch = pipeline_files.read_pipeline_product(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id
    )
    epoch_path = pipeline_files.get_product_path(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id
    )
    if epoch is None or epoch_path is None:
        print("Error loading epoch!")
        wait_for_key()
        return

    print(
        f"Starting analysis of {epoch_path.stem}: observation at {np.mean(epoch.HELIO)} AU"
    )

    stacking_method = StackingMethod.summation

    # load background-subtracted images
    uw1 = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uw1 is None or uvv is None:
        print("Error loading background-subtracted images!")
        wait_for_key()
        return

    # uw1_exposure_time = np.sum(epoch[epoch.FILTER == SwiftFilter.uw1].EXPOSURE)
    # uvv_exposure_time = np.sum(epoch[epoch.FILTER == SwiftFilter.uvv].EXPOSURE)

    df = surface_brightness_profiles(uw1=uw1, uvv=uvv, r_max=200)
    df = qh2o_from_surface_brightness_profiles(
        df=df, epoch=epoch, beta=beta_parameter(DustReddeningPercent(25))
    )
    print(df)

    # for c in ["cumulative_counts_uw1_mean", "cumulative_counts_uvv_mean"]:
    #     df[c].plot()
    # plt.show()
    #
    # for c in ["cumulative_counts_uw1_median", "cumulative_counts_uvv_median"]:
    #     df[c].plot()
    # plt.show()

    # for c in ["oh_brightness_running_total_median", "oh_brightness_running_total_mean"]:
    #     df[c].plot(legend=True)
    # plt.show()

    for c in ["flux_median", "flux_median_err"]:
        df[c].plot(legend=True)
    plt.show()

    # for c in ["flux_mean", "flux_mean_err"]:
    #     df[c].plot(legend=True)
    # plt.show()

    for c in ["qs_median", "qs_median_max", "qs_median_max_err"]:
        df[c].plot(legend=True)
    plt.show()

    wait_for_key()


def mark_comet_centers(swift_project_config: SwiftProjectConfig) -> None:
    """
    Finds images in uvv and uw1 filters, and outputs pngs images of each
    observation annotated with the center of the comet marked.
    Output images are placed in image_save_dir/[filter]/
    """

    pipeline_files = PipelineFiles(
        base_product_save_path=swift_project_config.product_save_path
    )

    swift_data = SwiftData(swift_project_config.swift_data_path)

    # select the epoch we want to process
    epoch_id = epoch_menu(pipeline_files=pipeline_files)
    if epoch_id is None:
        print("no epoch selected!")
        wait_for_key()
        return

    # load the epoch database
    epoch = pipeline_files.read_pipeline_product(
        PipelineProductType.epoch, epoch_id=epoch_id
    )
    if epoch is None:
        print("Error loading epoch!")
        wait_for_key()
        return

    image_save_dir: pathlib.Path = swift_project_config.product_save_path / "centers"
    plt.rcParams["figure.figsize"] = (15, 15)

    # directories to store the uw1 and uvv images: image_save_dir/[filter]/
    dir_by_filter = {
        SwiftFilter.uw1: image_save_dir
        / pathlib.Path(filter_to_file_string(SwiftFilter.uw1)),
        SwiftFilter.uvv: image_save_dir
        / pathlib.Path(filter_to_file_string(SwiftFilter.uvv)),
    }
    # create directories we will need if they don't exist
    for fdir in dir_by_filter.values():
        fdir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(epoch.iterrows(), total=epoch.shape[0])
    for _, row in progress_bar:
        obsid = row["OBS_ID"]
        extension = row["EXTENSION"]
        px = round(float(row["PX"]))
        py = round(float(row["PY"]))
        filter_type = row.FILTER
        filter_str = filter_to_file_string(filter_type)  # type: ignore

        # ask where the raw swift data FITS file is and read it
        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        output_image_name = pathlib.Path(f"{obsid}_{extension}_{filter_str}.png")
        output_image_path = dir_by_filter[filter_type] / output_image_name  # type: ignore

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_data)

        im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)  # type: ignore
        fig.colorbar(im1)
        # mark comet center
        plt.axvline(px, color="w", alpha=0.3)
        plt.axhline(py, color="w", alpha=0.3)

        ax1.set_title("C/2013US10")
        ax1.set_xlabel(f"{row['MID_TIME']}")
        ax1.set_ylabel(f"{row['FITS_FILENAME']}")

        plt.savefig(output_image_path)
        plt.close()
        # num_processed += 1

        progress_bar.set_description(f"Processed {obsid} extension {extension}")

    print("")
