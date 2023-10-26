#!/usr/bin/env python3

import os
import pathlib

import warnings
import sys
import logging as log

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

from swift_comet_pipeline.configs import read_swift_project_config
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.tui import get_selection
from swift_comet_pipeline.sun_direction import find_sun_direction


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
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


def main():
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    # load the config
    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)
    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    available_functions = ["sun_direction", "exit"]
    selection = available_functions[get_selection(available_functions)]
    print(selection)
    if selection == "sun_direction":
        find_sun_direction(swift_data=swift_data, pipeline_files=pipeline_files)
    else:
        return


if __name__ == "__main__":
    sys.exit(main())
