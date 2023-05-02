#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log


from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_types import (
    SwiftObservationLog,
    SwiftFilter,
    SwiftOrbitID,
    SwiftObservationID,
)
from swift_observation_log import match_by_orbit_ids_and_filters, read_observation_log
from stacking import includes_uvv_and_uw1_filters


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
    parser.add_argument(
        "observation_log_file", nargs=1, help="Filename of observation log input"
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


def test_OH_stackable(obs_log: SwiftObservationLog) -> None:
    has_both, _ = includes_uvv_and_uw1_filters(obs_log=obs_log)
    print(f"Data in uvv and uw1 filters in this range: {has_both}")


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    obs_log = read_observation_log(args.observation_log_file[0])

    # matching by orbit ids and filters: enough data that this should have both filters in dataset
    filter_types = SwiftFilter.all_filters()
    orbit_ids = list(
        map(SwiftOrbitID, ["00033759", "00033760", "00033822", "00033826", "00033827"])
    )
    print(f"Orbits: {orbit_ids}")
    ml1 = match_by_orbit_ids_and_filters(
        obs_log=obs_log, orbit_ids=orbit_ids, filter_types=filter_types
    )
    test_OH_stackable(ml1)

    obsid = SwiftObservationID("00033760001")
    print(f"Obsid: {obsid}")
    # this dataset should fail to have both filters
    ml2 = obs_log[obs_log["OBS_ID"] == obsid]
    test_OH_stackable(ml2)


if __name__ == "__main__":
    sys.exit(main())
