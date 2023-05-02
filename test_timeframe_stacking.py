#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import astropy.units as u

import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.time import Time
from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_types import (
    SwiftData,
    SwiftObservationLog,
    SwiftFilter,
    filter_to_string,
    SwiftStackingMethod,
)
from swift_observation_log import (
    read_observation_log,
    match_within_timeframe,
)
from stacking import (
    stack_image_by_selection,
    write_stacked_image,
    includes_uvv_and_uw1_filters,
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


def show_fits_scaled(image_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)

    im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    plt.show()


def test_stacking_by_selection(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    stacked_image_dir: pathlib.Path,
    stacking_method: SwiftStackingMethod,
) -> None:
    start_time = Time("2015-01-01T00:00:00.000")
    end_time = start_time + (20 * u.week)
    time_match = match_within_timeframe(
        obs_log=obs_log, start_time=start_time, end_time=end_time
    )

    (stackable, which_orbit_ids) = includes_uvv_and_uw1_filters(obs_log=time_match)
    if stackable:
        print("In the given time frame, these orbit ids have the necessary data:")
        for orbit_id in sorted(which_orbit_ids):
            print(f"\t{orbit_id}")
    else:
        print("The time frame given does not have data in both filters!")
        return

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        print(f"Stacking for filter {filter_to_string(filter_type)} ...")
        filter_mask = time_match["FILTER"] == filter_type
        ml = time_match[filter_mask]

        stacked_image = stack_image_by_selection(
            swift_data=swift_data,
            obs_log=ml,
            do_coincidence_correction=False,
            stacking_method=stacking_method,
        )

        if stacked_image is None:
            print("Stacking image failed :( ")
            return

        # show_fits_scaled(stacked_image.stacked_image)

        stacked_path, stacked_info_path = write_stacked_image(
            stacked_image_dir=stacked_image_dir, image=stacked_image
        )
        print(f"Wrote to {stacked_path}")
        print(f"Wrote info to {stacked_info_path}")


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

    for stacking_method in [SwiftStackingMethod.summation, SwiftStackingMethod.median]:
        test_stacking_by_selection(
            swift_data=swift_data,
            obs_log=obs_log,
            stacked_image_dir=stacked_image_dir,
            stacking_method=stacking_method,
        )


if __name__ == "__main__":
    sys.exit(main())
