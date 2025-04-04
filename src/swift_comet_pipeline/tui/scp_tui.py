#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log

import pandas as pd
from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

from rich.console import Console
from rich.text import Text
from rich import print as rprint

from swift_comet_pipeline.modeling.vectorial_model import vectorial_model_settings_init
from swift_comet_pipeline.modeling.vectorial_model_backend import VectorialModelBackend
from swift_comet_pipeline.modeling.vectorial_model_grid import VectorialModelGridQuality
from swift_comet_pipeline.orbits.orbital_data_download import download_orbital_data
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.read_swift_project_config import (
    read_swift_project_config,
)
from swift_comet_pipeline.projects.write_swift_project_config import (
    write_swift_project_config,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.tui.pipeline_extras import pipeline_extras_menu
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
from swift_comet_pipeline.tui.tui_common import clear_screen, get_yes_no
from swift_comet_pipeline.tui.tui_menus import step_status_to_symbol
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig

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


def read_or_create_project_config(
    swift_project_config_path: pathlib.Path,
) -> SwiftProjectConfig | None:
    # check if project config exists, and offer to create if not
    if not swift_project_config_path.exists():
        print(
            f"Config file {swift_project_config_path} does not exist! Would you like to create one now? (y/n)"
        )
        create_config = get_yes_no()
        if create_config:
            create_swift_project_config_from_input(
                swift_project_config_path=swift_project_config_path
            )
        else:
            return

    # load the project config
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return None

    return swift_project_config


def create_swift_project_config_from_input(
    swift_project_config_path: pathlib.Path,
) -> None:
    """
    Collect info on the data directories and how to identify the comet through JPL horizons,
    and write it to a yaml config
    """

    swift_data_path = pathlib.Path(input("Directory of the downloaded swift data: "))

    # try to validate that this path actually has data before accepting
    test_of_swift_data = SwiftData(data_path=swift_data_path)
    num_obsids = len(test_of_swift_data.get_all_observation_ids())
    if num_obsids == 0:
        rprint(
            "There doesn't seem to be data in the necessary format at [blue]{swift_data_path}[/blue]!"
        )
    else:
        rprint(
            f"Found appropriate data with a total of [green]{num_obsids}[/green] observation IDs"
        )

    project_path = pathlib.Path(
        input("Directory to store results and intermediate products: ")
    )

    jpl_horizons_id = input("JPL Horizons ID of the comet: ")

    # TODO: this fails on invalid input, make it more robust
    vm_quality = input(
        f"Vectorial model quality {VectorialModelGridQuality.all_qualities()}: "
    )

    # TODO: this fails on invalid input, make it more robust
    vm_backend = input(
        f"Vectorial model backend {VectorialModelBackend.all_model_backends()}: "
    )

    # TODO: finish questions for vectorial_fitting_requires_km and near_far_split_radius_km

    swift_project_config = SwiftProjectConfig(
        swift_data_path=swift_data_path,
        jpl_horizons_id=jpl_horizons_id,
        project_path=project_path,
        vectorial_model_quality=VectorialModelGridQuality(vm_quality),
        vectorial_model_backend=VectorialModelBackend(vm_backend),
        vectorial_fitting_requires_km=float(100_000),
        near_far_split_radius_km=float(50_000),
    )

    print(f"Writing project config to {swift_project_config_path}...")
    write_swift_project_config(
        config_path=swift_project_config_path, swift_project_config=swift_project_config
    )


def get_statuses(scp: SwiftCometPipeline) -> dict:
    status_map = {}
    for x in SwiftCometPipelineStepEnum.__members__.values():
        status_map[x] = f"{step_status_to_symbol(scp.get_overall_status(x))}"

    return status_map


# Description generator that appends the status
def generate_description(step: SwiftCometPipelineStepEnum, status_map: dict) -> str:
    descriptions = {
        SwiftCometPipelineStepEnum.observation_log: "Generate observation log",
        SwiftCometPipelineStepEnum.download_orbital_data: "Download orbital data",
        SwiftCometPipelineStepEnum.identify_epochs: "Identify epochs",
        SwiftCometPipelineStepEnum.veto_images: "Review and veto images",
        SwiftCometPipelineStepEnum.epoch_stack: "Stack images",
        SwiftCometPipelineStepEnum.determine_background: "Determine the background of stacked images",
        SwiftCometPipelineStepEnum.background_subtract: "Subtract background from stacked images",
        SwiftCometPipelineStepEnum.aperture_analysis: "Aperture analysis on the stacked images",
        SwiftCometPipelineStepEnum.vectorial_analysis: "Vectorial analysis on the stacked images",
        SwiftCometPipelineStepEnum.build_lightcurves: "Generate lightcurves",
        SwiftCometPipelineStepEnum.extra_functions: "Inspect pipeline status",
    }
    # Get the base description and append the status
    base_description = descriptions.get(step, "No description available")
    status = status_map[step]
    return f"{base_description} [Status: {status}]"


# Function to display the menu and let the user choose
def choose_pipeline_step(
    swift_project_config: SwiftProjectConfig,
) -> SwiftCometPipelineStepEnum | None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    console = Console()
    status_map = get_statuses(scp)

    # dictionary that maps descriptions to the enum values
    description_to_pipeline_step = {
        generate_description(step, status_map): step
        for step in SwiftCometPipelineStepEnum
    }

    exit_option = "Exit"
    descriptions = list(description_to_pipeline_step.keys()) + [exit_option]

    console.print(Text("Select a pipeline step:", style="bold magenta"))
    for i, description in enumerate(descriptions, 1):
        console.print(f"[bold]{i}.[/bold] {description}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(descriptions):
                selected_description = descriptions[choice - 1]
                if selected_description == exit_option:
                    console.print("[yellow]Exiting the menu...[/yellow]")
                    return None
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
        print("Could not load a valid project configuration! Exiting.")
        return 1

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
            case SwiftCometPipelineStepEnum.extra_functions:
                pipeline_extras_menu(swift_project_config=swift_project_config)
            case None:
                exit_program = True


if __name__ == "__main__":
    sys.exit(main())
