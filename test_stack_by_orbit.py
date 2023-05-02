#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log

from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_types import (
    SwiftData,
    SwiftObservationLog,
    SwiftFilter,
    SwiftOrbitID,
    filter_to_string,
    SwiftStackingMethod,
    SwiftPixelResolution,
)
from swift_observation_log import read_observation_log
from stacking import stack_image_by_selection, write_stacked_image


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


def test_stacking_by_orbit(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    stacked_image_dir: pathlib.Path,
    stacking_method: SwiftStackingMethod,
) -> None:
    # select an orbit
    orbit_id = SwiftOrbitID("00034423")
    mask = obs_log["ORBIT_ID"] == orbit_id

    ml = obs_log[mask]

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        print(f"Stacking for filter {filter_to_string(filter_type)} ...")
        filter_mask = ml["FILTER"] == filter_type
        match_by_filter = ml[filter_mask]

        if len(match_by_filter) == 0:
            print(
                f"No data found for orbit {orbit_id} in filter {filter_to_string(filter_type)}: skipping."
            )
            continue

        stacked_image = stack_image_by_selection(
            swift_data=swift_data,
            obs_log=match_by_filter,
            do_coincidence_correction=False,
            detector_scale=SwiftPixelResolution.data_mode,
            stacking_method=stacking_method,
        )

        if stacked_image is None:
            print("Stacking image failed :( ")
            return

        stacked_path, stacked_info_path = write_stacked_image(
            stacked_image_dir=stacked_image_dir, image=stacked_image
        )
        print(f"Wrote to {stacked_path}, info at {stacked_info_path}")


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1
    stacked_image_dir = swift_config["stacked_image_dir"]

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    obs_log = read_observation_log(args.observation_log_file[0])

    test_stacking_by_orbit(
        swift_data=swift_data,
        obs_log=obs_log,
        stacked_image_dir=stacked_image_dir,
        stacking_method=SwiftStackingMethod.median,
    )


if __name__ == "__main__":
    sys.exit(main())
