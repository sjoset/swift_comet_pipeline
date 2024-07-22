from itertools import product
from rich import print as rprint
from rich.panel import Panel
from icecream import ic

from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.tui.tui_common import bool_to_x_or_check


def pipeline_extra_status(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)

    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    print("")

    rprint(
        "[orange3]Observation log:[/orange3] ",
        bool_to_x_or_check(data_ingestion_files.observation_log.exists()),
    )
    rprint(
        "[orange3]Comet orbit data:[/orange3] ",
        bool_to_x_or_check(data_ingestion_files.comet_orbital_data.exists()),
    )
    rprint(
        "[orange3]Earth orbit data:[/orange3] ",
        bool_to_x_or_check(data_ingestion_files.earth_orbital_data.exists()),
    )

    print("")

    epoch_products = data_ingestion_files.epochs
    if epoch_products is None:
        print("No epochs defined yet!")
        return

    print("")
    for parent_epoch in epoch_products:

        epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
            parent_epoch=parent_epoch
        )
        if epoch_subpipeline is None:
            ic(
                f"Cannot create epoch subpipeline for {parent_epoch.epoch_id}! Skipping."
            )
            continue

        rprint(Panel(f"Epoch {parent_epoch.epoch_id}:", expand=False))
        rprint("[orange3]Stacked images:[/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            stack_exists = epoch_subpipeline.stacked_images[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(stack_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Background analysis:[/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            bg_exists = epoch_subpipeline.background_analyses[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(bg_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Background-subtracted images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            img_exists = epoch_subpipeline.background_subtracted_images[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Q vs aperture radius analysis: [/orange3]")
        for stacking_method in stacking_methods:
            q_vs_aperture_exists = epoch_subpipeline.qh2o_vs_aperture_radius_analyses[
                stacking_method
            ].product_path.exists()
            rprint(
                f"{stacking_method} {bool_to_x_or_check(q_vs_aperture_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Radial profile extraction: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            extracted_profile_exists = epoch_subpipeline.extracted_profiles[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(extracted_profile_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Extracted profile images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            extracted_profile_img_exists = epoch_subpipeline.extracted_profile_images[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(extracted_profile_img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Median subtracted images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            med_sub_img_exists = epoch_subpipeline.median_subtracted_images[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(med_sub_img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Median divided images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            med_div_img_exists = epoch_subpipeline.median_divided_images[
                filter_type, stacking_method
            ].product_path.exists()
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(med_div_img_exists)}\t",
                end="",
            )
        print("\n\n")
