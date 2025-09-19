#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log
from argparse import ArgumentParser

import pandas as pd
from astropy.wcs.wcs import FITSFixedWarning
from rich import print as rprint

from swift_comet_pipeline.project_configuration.read_swift_comet_project_config import (
    read_swift_comet_project_config,
)
from swift_comet_pipeline.scp_types.compound.swift_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.tui.tui_common import get_yes_no


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
) -> CometProjectConfig | None:
    # check if project config exists, and offer to create if not
    if not swift_project_config_path.exists():
        print(
            f"Config file {swift_project_config_path} does not exist! Would you like to create one now? (y/n)"
        )
        return None
        # create_config = get_yes_no()
        # if create_config:
        #     create_swift_project_config_from_input(
        #         swift_project_config_path=swift_project_config_path
        #     )
        # else:
        #     return

    # load the project config
    swift_project_config = read_swift_comet_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return None

    return swift_project_config


# # TODO: fix this function
# def create_swift_project_config_from_input(
#     swift_project_config_path: pathlib.Path,
# ) -> None:
#     """
#     Collect info on the data directories and how to identify the comet through JPL horizons,
#     and write it to a yaml config
#     """
#
#     swift_data_path = pathlib.Path(input("Directory of the downloaded swift data: "))
#
#     # try to validate that this path actually has data before accepting
#     test_of_swift_data = SwiftData(data_path=swift_data_path)
#     num_obsids = len(test_of_swift_data._find_all_observation_ids())
#     if num_obsids == 0:
#         rprint(
#             "There doesn't seem to be data in the necessary format at [blue]{swift_data_path}[/blue]!"
#         )
#     else:
#         rprint(
#             f"Found appropriate data with a total of [green]{num_obsids}[/green] observation IDs"
#         )
#
#     project_path = pathlib.Path(
#         input("Directory to store results and intermediate products: ")
#     )
#
#     jpl_horizons_id = input("JPL Horizons ID of the comet: ")
#
#     # TODO: this fails on invalid input, make it more robust
#     vm_quality = input(
#         f"Vectorial model quality {VectorialModelGridQuality.all_qualities()}: "
#     )
#
#     # TODO: this fails on invalid input, make it more robust
#     vm_backend = input(
#         f"Vectorial model backend {VectorialModelBackend.all_model_backends()}: "
#     )
#
#     # TODO: finish questions for vectorial_fitting_requires_km and near_far_split_radius_km
#
#     swift_project_config = SwiftProjectConfig(
#         swift_data_path=swift_data_path,
#         jpl_horizons_id=jpl_horizons_id,
#         project_path=project_path,
#         vectorial_model_quality=VectorialModelGridQuality(vm_quality),
#         vectorial_model_backend=VectorialModelBackend(vm_backend),
#         vectorial_fitting_requires_km=float(100_000),
#         near_far_split_radius_km=float(50_000),
#     )
#
#     print(f"Writing project config to {swift_project_config_path}...")
#     write_swift_project_config(
#         config_path=swift_project_config_path, swift_project_config=swift_project_config
#     )


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 0)

    args = process_args()
    swift_comet_project_config_path = pathlib.Path(args.swift_project_config)

    comet_project_config = read_or_create_project_config(
        swift_project_config_path=swift_comet_project_config_path
    )

    if comet_project_config is None:
        print("Could not load a valid project configuration! Exiting.")
        return 1

    # set up the cache db and other stuff
    vectorial_model_settings_init(swift_project_config=comet_project_config)

    # TODO: add option to view stacked image pairs

    exit_program = False
    while not exit_program:
        clear_screen()
        step = choose_pipeline_step(comet_project_config)
        match step:
            case SwiftCometPipelineStepEnum.observation_log:
                observation_log_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.download_orbital_data:
                download_orbital_data(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.identify_epochs:
                identify_epochs_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.veto_images:
                veto_epoch_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.epoch_stack:
                epoch_stacking_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.determine_background:
                determine_background_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.background_subtract:
                background_subtract_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.aperture_analysis:
                aperture_analysis_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.vectorial_analysis:
                vectorial_analysis_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.build_lightcurves:
                build_lightcurves_step(swift_project_config=comet_project_config)
            case SwiftCometPipelineStepEnum.extra_functions:
                pipeline_extras_menu(swift_project_config=comet_project_config)
            case None:
                exit_program = True


if __name__ == "__main__":
    sys.exit(main())
