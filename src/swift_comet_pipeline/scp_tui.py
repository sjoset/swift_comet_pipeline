#!/usr/bin/env python3

import os
import pathlib
import sys
from typing import Optional
import warnings

import logging as log

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.configs import (
    SwiftProjectConfig,
    read_swift_project_config,
    write_swift_project_config,
)
from swift_comet_pipeline.observation_log import build_observation_log
from swift_comet_pipeline.tui import get_yes_no


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
) -> Optional[SwiftProjectConfig]:
    # check if project config exists, and offer to create if not
    if not swift_project_config_path.exists():
        print(
            f"Config file {swift_project_config_path} does not exist! Would you like to create one now?"
        )
        create_config = get_yes_no()
        if create_config:
            create_swift_project_config(
                swift_project_config_path=swift_project_config_path
            )
        else:
            print("Ok, exiting.")
            return

    # load the project config
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return None

    return swift_project_config


def create_swift_project_config(swift_project_config_path: pathlib.Path) -> None:
    """
    Collect info on the data directories and how to identify the comet through JPL horizons,
    and write it to a yaml config
    """

    print(
        f"Creating project config {swift_project_config_path}\n-----------------------"
    )

    swift_data_path = pathlib.Path(input("Directory of the downloaded swift data: "))

    # try to validate that this path actually has data before accepting
    test_of_swift_data = SwiftData(data_path=swift_data_path)
    num_obsids = len(test_of_swift_data.get_all_observation_ids())
    if num_obsids == 0:
        print(
            "There doesn't seem to be data in the necessary format at {swift_data_path}!"
        )
    else:
        print(f"Found appropriate data with a total of {num_obsids} observation IDs")

    product_save_path = pathlib.Path(
        input("Directory to store results and intermediate products: ")
    )

    jpl_horizons_id = input("JPL Horizons ID of the comet: ")

    swift_project_config = SwiftProjectConfig(
        swift_data_path=swift_data_path,
        product_save_path=product_save_path,
        jpl_horizons_id=jpl_horizons_id,
    )

    write_swift_project_config(
        config_path=swift_project_config_path, swift_project_config=swift_project_config
    )


def obs_log_main(swift_project_config: SwiftProjectConfig):
    horizons_id = swift_project_config.jpl_horizons_id
    sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    # TODO: check if observation log exists and ask user whether to exit or generate new and overwrite

    df = build_observation_log(
        swift_data=sdd,
        obsids=sdd.get_all_observation_ids(),
        horizons_id=horizons_id,
    )

    if df is None:
        print(
            "Could not construct the observation log in memory, exiting without writing output."
        )
        return 1

    pipeline_files.observation_log.data_product = df
    pipeline_files.observation_log.save_product()


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()
    swift_project_config_path = pathlib.Path(args.swift_project_config)

    swift_project_config = read_or_create_project_config(
        swift_project_config_path=swift_project_config_path
    )
    print(swift_project_config)


if __name__ == "__main__":
    sys.exit(main())
