#!/usr/bin/env python3

import sys
import os
import pathlib
import numpy as np
import logging as log

from argparse import ArgumentParser
from astroquery.jplhorizons import Horizons

from read_swift_config import read_swift_config
from swift_observation_log import read_observation_log


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


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    horizon_id = swift_config["jpl_horizons_id"]

    obs_log = read_observation_log(args.observation_log_file[0])

    time_start = np.min(obs_log["MID_TIME"])
    time_stop = np.max(obs_log["MID_TIME"])

    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "10d"}

    horizons_response = Horizons(
        id="C/2013 US10", location="@Geocenter", id_type="smallbody", epochs=epochs
    )
    vectors = horizons_response.vectors()

    df = vectors.to_pandas()
    df.to_csv("horizons_orbital_data.csv")


if __name__ == "__main__":
    sys.exit(main())
