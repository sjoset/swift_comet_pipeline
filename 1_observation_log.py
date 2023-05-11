#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

from swift_types import SwiftData
from read_swift_config import read_swift_config
from swift_observation_log import build_observation_log

__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="YAML configuration file to use"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="observation_log.csv",
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
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    horizon_id = swift_config["jpl_horizons_id"]

    sdd = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    df = build_observation_log(
        swift_data=sdd, obsids=sdd.get_all_observation_ids(), horizon_id=horizon_id
    )

    if df is None:
        print(
            "Could not construct the observation log in memory, exiting without writing output."
        )
        return 1

    # write out our dataframe
    df.to_csv(args.output)


if __name__ == "__main__":
    sys.exit(main())
