from itertools import product

import questionary
import numpy as np
from astropy.io import fits
from icecream import ic
from rich import print as rprint

from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.background.background_result import (
    BackgroundResult,
    background_result_to_dict,
)
from swift_comet_pipeline.background.determine_background import determine_background
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.tui.tui_common import (
    stacked_epoch_menu,
)


def get_background_method_choice() -> BackgroundDeterminationMethod | None:

    bg_method = questionary.select(
        message="Background method: ",
        choices=BackgroundDeterminationMethod.all_bg_determination_methods(),
    ).ask()

    return bg_method


def get_background(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    filter_type: SwiftFilter,
    bg_method: BackgroundDeterminationMethod,
    helio_r_au: float,
) -> BackgroundResult:

    bg_cr = determine_background(
        img=img,
        filter_type=filter_type,
        exposure_map=exposure_map,
        background_method=bg_method,
        helio_r_au=helio_r_au,
    )

    return bg_cr


def background_analysis_step(swift_project_config: SwiftProjectConfig) -> None:
    uw1_and_uvv = [SwiftFilter.uw1, SwiftFilter.uvv]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)

    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    selected_parent_epoch = stacked_epoch_menu(
        pipeline_files=pipeline_files,
        require_background_analysis_to_exist=False,
        require_background_analysis_to_not_exist=True,
    )
    if selected_parent_epoch is None:
        print("Could not select parent epoch, exiting.")
        return

    epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        parent_epoch=selected_parent_epoch
    )
    if epoch_subpipeline is None:
        ic(f"No subpipeline for epoch {selected_parent_epoch.epoch_id}! This is a bug.")
        return

    stacked_image_set = epoch_subpipeline.get_stacked_image_set()
    if stacked_image_set is None:
        ic(
            f"Could not load stacked image set for epoch {selected_parent_epoch.epoch_id}!"
        )
        return

    bg_method = get_background_method_choice()
    if bg_method is None:
        return

    epoch_subpipeline.stacked_epoch.read_product_if_not_loaded()
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        print("Error reading stacked epoch!")
        return
    helio_r_au = np.mean(stacked_epoch.HELIO)

    for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
        img_data = stacked_image_set[filter_type, stacking_method]
        img_header = epoch_subpipeline.stacked_images[
            filter_type, stacking_method
        ].data.header

        epoch_subpipeline.exposure_map[filter_type].read_product_if_not_loaded()
        exposure_map = epoch_subpipeline.exposure_map[filter_type].data.data

        bg_result = get_background(
            img_data,
            exposure_map=exposure_map,
            filter_type=filter_type,
            bg_method=bg_method,
            helio_r_au=helio_r_au,
        )

        print(f"{epoch_subpipeline.parent_epoch.epoch_id}")
        print(f"Background count rate: {bg_result.count_rate_per_pixel}")

        rprint(
            f"[green]Writing background analysis for filter {filter_to_file_string(filter_type)}, stacking method {stacking_method}...[/green]"
        )
        epoch_subpipeline.background_analyses[filter_type, stacking_method].data = (
            background_result_to_dict(bg_result=bg_result)
        )
        epoch_subpipeline.background_analyses[filter_type, stacking_method].write()

        bg_corrected_img = img_data - bg_result.count_rate_per_pixel.value

        # make a new fits with the background-corrected image, and copy the header information over from the original stacked image
        rprint(
            f"[green]Writing background-subtracted FITS image for filter {filter_to_file_string(filter_type)}, stacking method {stacking_method}...[/green]"
        )
        epoch_subpipeline.background_subtracted_images[
            filter_type, stacking_method
        ].data = fits.ImageHDU(data=bg_corrected_img, header=img_header)
        epoch_subpipeline.background_subtracted_images[
            filter_type, stacking_method
        ].write()
