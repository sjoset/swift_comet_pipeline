#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser
from pipeline_files import PipelineFiles

from swift_data import SwiftData
from configs import read_swift_project_config
from observation_log import build_observation_log


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config", nargs=1, help="Filename of project config"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="observation_log.parquet",
        help="Filename of observation log output",
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
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    # load the project config
    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return 1
    horizons_id = swift_project_config.jpl_horizons_id

    # the raw swift data lives here
    sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    # put together the pipeline file list
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

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


if __name__ == "__main__":
    sys.exit(main())
