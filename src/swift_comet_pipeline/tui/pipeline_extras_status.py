from itertools import product
from rich import print as rprint
from rich.panel import Panel

from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps import (
    SwiftCometPipelineStepStatus,
)
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.tui.tui_common import bool_to_x_or_check


def pipeline_extra_status(swift_project_config: SwiftProjectConfig) -> None:

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    print("")

    rprint(
        "[orange3]Observation log:[/orange3] ",
        bool_to_x_or_check(scp.exists(pf=PipelineFilesEnum.observation_log)),
    )
    rprint(
        "[orange3]Comet orbit data:[/orange3] ",
        bool_to_x_or_check(scp.exists(pf=PipelineFilesEnum.comet_orbital_data)),
    )
    rprint(
        "[orange3]Earth orbit data:[/orange3] ",
        bool_to_x_or_check(scp.exists(pf=PipelineFilesEnum.earth_orbital_data)),
    )

    print("")

    if (
        scp.get_status(step=SwiftCometPipelineStepEnum.identify_epochs)
        != SwiftCometPipelineStepStatus.complete
    ):
        print("No epochs defined yet!")
        return

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None

    print("")
    for epoch_id in epoch_ids:

        rprint(Panel(f"Epoch {epoch_id}:", expand=False))
        rprint("[orange3]Stacked images:[/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            stack_exists = scp.exists(
                pf=PipelineFilesEnum.stacked_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(stack_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Background determination:[/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            bg_det_exists = scp.exists(
                pf=PipelineFilesEnum.background_determination,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(bg_det_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Background-subtracted images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            img_exists = scp.exists(
                pf=PipelineFilesEnum.background_subtracted_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Q vs aperture radius analysis: [/orange3]")
        for stacking_method in stacking_methods:
            ap_analysis_exists = scp.exists(
                pf=PipelineFilesEnum.aperture_analysis,
                epoch_id=epoch_id,
                stacking_method=stacking_method,
            )
            rprint(
                f"{stacking_method} {bool_to_x_or_check(ap_analysis_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Radial profile extraction: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            ex_prof_exists = scp.exists(
                pf=PipelineFilesEnum.extracted_profile,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(ex_prof_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Extracted profile images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            ex_prof_img_exists = scp.exists(
                pf=PipelineFilesEnum.extracted_profile_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(ex_prof_img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Median subtracted images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            med_sub_img_exists = scp.exists(
                PipelineFilesEnum.median_subtracted_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(med_sub_img_exists)}\t",
                end="",
            )
        print("")

        rprint("[orange3]Median divided images: [/orange3]")
        for filter_type, stacking_method in product(filters, stacking_methods):
            med_div_img_exists = scp.exists(
                pf=PipelineFilesEnum.median_divided_image,
                epoch_id=epoch_id,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )
            rprint(
                f"{filter_to_file_string(filter_type)} {stacking_method}: {bool_to_x_or_check(med_div_img_exists)}\t",
                end="",
            )
        print("\n\n")
