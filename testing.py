#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_data import SwiftData
from swift_types import (
    SwiftObservationLog,
    SwiftFilter,
    swift_observation_id_from_int,
    filter_to_obs_string,
)
from swift_observation_log import get_observation_log_rows_that_match


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


def test_swift_data(sdd: SwiftData) -> None:
    print(sdd.get_all_orbit_ids())


def test_observation_log(obs_log: SwiftObservationLog) -> None:
    for obsid_int in np.unique(obs_log["OBS_ID"]):
        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            obsid = swift_observation_id_from_int(obsid_int)
            if obsid is None:
                continue

            newdf = get_observation_log_rows_that_match(
                obs_log,
                obsid,
                filter_type=filter_type,
            )

            for _, row in newdf.iterrows():
                print(f"{row['OBS_ID']} | {row['EXTENSION']} | {row['FILTER']}")


# def test_image_resizing_method(obsids: List[SwiftObservationID]) -> None:
#     for obsid in obsids:
#         new_image_size = determine_image_resize_dimensions_by_obsid(
#             swift_data=sdd,
#             obs_log=obs_log,
#             obsid=obsid,
#             filter_type=SwiftFilterObsString.uw1,
#         )


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    sdd = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    test_swift_data(sdd)

    obs_log = pd.read_csv(args.observation_log_file[0])
    test_observation_log(obs_log)


if __name__ == "__main__":
    sys.exit(main())
