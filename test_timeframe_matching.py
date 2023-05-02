#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import astropy.units as u

from astropy.time import Time
from argparse import ArgumentParser

from swift_types import SwiftObservationLog
from read_swift_config import read_swift_config
from swift_observation_log import (
    read_observation_log,
    match_within_timeframe,
)
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


def test_timeframe_matching(obs_log: SwiftObservationLog) -> None:
    start_time = Time("2014-12-19T00:27:21.000")
    # end_time = Time("2015-01-01T00:00:00.000")
    end_time = start_time + (8 * u.week)

    print(f"Matching observations between {start_time} through {end_time}:")
    ml = match_within_timeframe(
        obs_log=obs_log, start_time=start_time, end_time=end_time
    )

    ts = ml["MID_TIME"].values
    ts.sort()

    (stackable, which_orbit_ids) = includes_uvv_and_uw1_filters(obs_log=ml)
    if stackable:
        print("In the given time frame, these orbit ids have relevant data:")
        for orbit_id in sorted(which_orbit_ids):
            print(f"\t{orbit_id}")
    else:
        print("The time frame given does not have data in both filters!")


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    obs_log = read_observation_log(args.observation_log_file[0])

    test_timeframe_matching(obs_log=obs_log)


if __name__ == "__main__":
    sys.exit(main())
