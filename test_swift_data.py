#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import numpy as np
import astropy.units as u

# import pandas as pd
import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits
from astropy.time import Time
from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_types import (
    SwiftData,
    SwiftObservationLog,
    SwiftObservationID,
    SwiftFilter,
    SwiftOrbitID,
    filter_to_string,
    SwiftStackingMethod,
)
from swift_observation_log import (
    match_by_obsids_and_filters,
    # match_by_orbit_ids_and_filters,
    read_observation_log,
    get_obsids_in_orbits,
    match_within_timeframe,
)
from stacking import (
    stack_image_by_selection,
    write_stacked_image,
)


__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
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

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def test_swift_data(sdd: SwiftData) -> None:
    print("SwiftData check: listing all orbits ...")
    print(sorted(sdd.get_all_orbit_ids()))
    print("SwiftData check: listing all obsids ...")
    print(sorted(sdd.get_all_observation_ids()))


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )
    test_swift_data(swift_data)


if __name__ == "__main__":
    sys.exit(main())
