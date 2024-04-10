from itertools import product
from rich import print as rprint
from rich.panel import Panel

from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.tui.tui_common import bool_to_x_or_check, wait_for_key
from swift_comet_pipeline.pipeline.pipeline_files import (
    PipelineFiles,
    PipelineProductType,
)


def pipeline_extra_status(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    print("")

    rprint(
        "[orange3]Observation log:[/orange3] ",
        bool_to_x_or_check(pipeline_files.exists(PipelineProductType.observation_log)),
    )
    rprint(
        "[orange3]Comet orbit data:[/orange3] ",
        bool_to_x_or_check(
            pipeline_files.exists(PipelineProductType.comet_orbital_data)
        ),
    )
    rprint(
        "[orange3]Earth orbit data:[/orange3] ",
        bool_to_x_or_check(
            pipeline_files.exists(PipelineProductType.earth_orbital_data)
        ),
    )

    print("")

    epoch_ids = pipeline_files.get_epoch_ids()
    if epoch_ids is None:
        print("No epochs defined yet!")
        wait_for_key()
        return

    print("")
    for epoch_id in epoch_ids:
        epoch_path = pipeline_files.get_product_path(
            PipelineProductType.epoch, epoch_id=epoch_id
        )
        if epoch_path is None:
            continue

        rprint(Panel(f"Epoch {epoch_path.stem}:", expand=False))
        print("Stacked images:")
        for filter_type, stacking_method in product(filters, stacking_methods):
            # does the stacked image associated with this stacked epoch exist?
            stack_exists = pipeline_files.exists(
                PipelineProductType.stacked_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(stack_exists)}\t",
                end="",
            )
        print("")

        print("Background analysis:")
        for filter_type, stacking_method in product(filters, stacking_methods):
            bg_exists = pipeline_files.exists(
                PipelineProductType.background_analysis,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(bg_exists)}\t",
                end="",
            )
        print("")

        print("Background-subtracted images: ")
        for filter_type, stacking_method in product(filters, stacking_methods):
            img_exists = pipeline_files.exists(
                PipelineProductType.background_subtracted_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(img_exists)}\t",
                end="",
            )
        print("")

        print("Q vs aperture radius analysis: ")
        for stacking_method in stacking_methods:
            q_exists = pipeline_files.exists(
                PipelineProductType.qh2o_vs_aperture_radius,
                epoch_id=epoch_id,
                stacking_method=stacking_method,
            )
            rprint(f"{stacking_method} {bool_to_x_or_check(q_exists)}\t", end="")
        print("\n\n")

    wait_for_key()
