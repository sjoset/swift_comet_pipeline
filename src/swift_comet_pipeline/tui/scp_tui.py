#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log

import pandas as pd
from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

# import questionary
from rich.console import Console
from rich.text import Text

from swift_comet_pipeline.modeling.vectorial_model import vectorial_model_settings_init
from swift_comet_pipeline.orbits.orbital_data_download import download_orbital_data
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.configs import (
    read_or_create_project_config,
)
from swift_comet_pipeline.projects.swift_project_config import SwiftProjectConfig
from swift_comet_pipeline.tui.pipeline_steps_aperture_analysis import (
    aperture_analysis_step,
)
from swift_comet_pipeline.tui.pipeline_steps_background_analysis import (
    background_subtract_step,
    determine_background_step,
)
from swift_comet_pipeline.tui.pipeline_steps_build_lightcurves import (
    build_lightcurves_step,
)
from swift_comet_pipeline.tui.pipeline_steps_observation_log import (
    observation_log_step,
)
from swift_comet_pipeline.tui.pipeline_steps_identify_epochs import identify_epochs_step
from swift_comet_pipeline.tui.pipeline_steps_vectorial_analysis import (
    vectorial_analysis_step,
)
from swift_comet_pipeline.tui.pipeline_steps_veto_epoch import veto_epoch_step
from swift_comet_pipeline.tui.pipeline_steps_epoch_stacking import epoch_stacking_step
from swift_comet_pipeline.tui.tui_common import clear_screen
from swift_comet_pipeline.tui.tui_menus import step_status_to_symbol

# class PipelineStepsMenuEntry(StrEnum):
#     observation_log = "generate observation log"
#     identify_epochs = "slice observation log into epochs"
#     veto_epoch = "view and veto images in epoch"
#     epoch_stacking = "stack images in an epoch"
#     background_analysis = "background analysis and subtraction"
#     qH2O_vs_aperture_radius = "water production as a function of aperture radius"
#     qH2O_from_profile = "water production from a comet profile/slice"
#     vectorial_fitting = (
#         "derive water production from comet profile via fitting to model"
#     )
#     generate_lightcurve = "generate lightcurve"
#
#     extra_functions = "extra functions"
#     exit_menu = "exit menu"
#
#     @classmethod
#     def all_pipeline_steps(cls):
#         return [x for x in cls]


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config",
        nargs="?",
        help="Filename of project config",
        default="config.yaml",
    )

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


# # Example description generator for each enum entry
# def generate_description(step: SwiftCometPipelineStep) -> str:
#     descriptions = {
#         SwiftCometPipelineStep.observation_log: "Log observations from the Swift comet.",
#         SwiftCometPipelineStep.identify_epochs: "Identify key epochs in the observation data.",
#         SwiftCometPipelineStep.veto_images: "Review and veto specific images.",
#         SwiftCometPipelineStep.download_orbital_data: "Download orbital data for further analysis.",
#         SwiftCometPipelineStep.determine_background: "Determine the background for epoch analysis.",
#         SwiftCometPipelineStep.epoch_stack: "Stack images from a given epoch.",
#         SwiftCometPipelineStep.aperture_analysis: "Perform aperture analysis on the stacked images.",
#         SwiftCometPipelineStep.vectorial_analysis: "Run vectorial analysis on the comet images.",
#         SwiftCometPipelineStep.build_lightcurves: "Generate lightcurves based on the analysis."
#     }
#     return descriptions.get(step, "No description available")
#
#
# def select_pipeline_step(
#     swift_project_config: SwiftProjectConfig,
# ) -> PipelineStepsMenuEntry | None:
#
#     # Create the reverse dictionary where descriptions are keys
#     description_to_menu_entry = {
#         generate_description(entry): entry for entry in PipelineStepsMenuEntry
#     }
#
#     # Ask the user to choose an option, directly using the descriptions
#     selected_description = questionary.select(
#         "Select a pipeline step:",
#         choices=list(
#             description_to_menu_entry.keys()
#         ),  # Present descriptions to the user
#     ).ask()
#
#     # Use the selected description to find and return the corresponding PipelineStepsMenuEntry
#     return description_to_menu_entry.get(selected_description)


def get_statuses(scp: SwiftCometPipeline) -> dict:
    status_map = {}
    for x in SwiftCometPipelineStepEnum.__members__.values():
        status_map[x] = f"{step_status_to_symbol(scp.get_overall_status(x))}"

    return status_map


# Description generator that appends the status
def generate_description(step: SwiftCometPipelineStepEnum, status_map: dict) -> str:
    descriptions = {
        SwiftCometPipelineStepEnum.observation_log: "Log observations from the Swift comet.",
        SwiftCometPipelineStepEnum.identify_epochs: "Identify key epochs in the observation data.",
        SwiftCometPipelineStepEnum.veto_images: "Review and veto specific images.",
        SwiftCometPipelineStepEnum.download_orbital_data: "Download orbital data for further analysis.",
        SwiftCometPipelineStepEnum.epoch_stack: "Stack images from a given epoch.",
        SwiftCometPipelineStepEnum.determine_background: "Determine the background for epoch analysis.",
        SwiftCometPipelineStepEnum.background_subtract: "Subtract background from stacked images",
        SwiftCometPipelineStepEnum.aperture_analysis: "Perform aperture analysis on the stacked images.",
        SwiftCometPipelineStepEnum.vectorial_analysis: "Run vectorial analysis on the comet images.",
        SwiftCometPipelineStepEnum.build_lightcurves: "Generate lightcurves based on the analysis.",
    }
    # Get the base description and append the status
    base_description = descriptions.get(step, "No description available")
    status = status_map[step]
    # print(f"{step} --> {base_description}")
    return f"{base_description} [Status: {status}]"


# Function to display the menu and let the user choose
def choose_pipeline_step(
    swift_project_config: SwiftProjectConfig,
) -> SwiftCometPipelineStepEnum | None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    console = Console()
    status_map = get_statuses(scp)

    # Create a dictionary that maps descriptions to the enum values
    description_to_pipeline_step = {
        generate_description(step, status_map): step
        for step in SwiftCometPipelineStepEnum
    }

    # Add a custom 'Exit' option
    exit_option = "Exit"
    descriptions = list(description_to_pipeline_step.keys()) + [exit_option]

    # Print the menu using rich
    console.print(Text("Select a pipeline step:", style="bold magenta"))
    for i, description in enumerate(descriptions, 1):
        console.print(f"[bold]{i}.[/bold] {description}")

    # Prompt the user to select an option
    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(descriptions):
                selected_description = descriptions[choice - 1]
                if selected_description == exit_option:
                    console.print("[yellow]Exiting the menu...[/yellow]")
                    return None  # Or handle the exit logic as needed
                else:
                    return description_to_pipeline_step.get(selected_description)
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 0)

    args = process_args()
    swift_project_config_path = pathlib.Path(args.swift_project_config)

    swift_project_config = read_or_create_project_config(
        swift_project_config_path=swift_project_config_path
    )

    if swift_project_config is None:
        print("Could not load a valid configuration! Exiting.")
        return

    # set up the cache db and other stuff
    vectorial_model_settings_init(swift_project_config=swift_project_config)

    # TODO: add option to view stacked image pairs

    exit_program = False
    while not exit_program:
        clear_screen()
        step = choose_pipeline_step(swift_project_config)
        match step:
            case SwiftCometPipelineStepEnum.observation_log:
                observation_log_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.download_orbital_data:
                download_orbital_data(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.identify_epochs:
                identify_epochs_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.veto_images:
                veto_epoch_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.epoch_stack:
                epoch_stacking_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.determine_background:
                determine_background_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.background_subtract:
                background_subtract_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.aperture_analysis:
                aperture_analysis_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.vectorial_analysis:
                vectorial_analysis_step(swift_project_config=swift_project_config)
            case SwiftCometPipelineStepEnum.build_lightcurves:
                build_lightcurves_step(swift_project_config=swift_project_config)
            case None:
                exit_program = True

        # elif step == SwiftCometPipelineStepEnum.extra_functions:
        #     pipeline_extras_menu(swift_project_config=swift_project_config)
        # else:
        #     exit_program = True
        # wait_for_key()


if __name__ == "__main__":
    sys.exit(main())
