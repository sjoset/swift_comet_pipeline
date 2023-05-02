#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
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
    SwiftFilter,
    filter_to_string,
)
from swift_observation_log import (
    # match_by_orbit_ids_and_filters,
    read_observation_log,
    match_within_timeframe,
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


def mark_comet_centers(swift_data: SwiftData, obs_log: SwiftObservationLog) -> None:
    plt.rcParams["figure.figsize"] = (15, 15)
    for _, row in obs_log.iterrows():
        obsid = row["OBS_ID"]
        extension = row["EXTENSION"]
        px = round(float(row["PX"]))
        py = round(float(row["PY"]))
        filter_str = filter_to_string(row["FILTER"])

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        output_image_path = pathlib.Path(f"{obsid}_{extension}_{filter_str}.png")

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_data)

        im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)
        fig.colorbar(im1)
        # mark comet center
        plt.axvline(px, color="w", alpha=0.3)
        plt.axhline(py, color="w", alpha=0.3)

        ax1.set_title("Catalina")
        ax1.set_xlabel(f"{row['MID_TIME']}")
        ax1.set_ylabel(f"{row['FITS_FILENAME']}")

        plt.savefig(output_image_path)
        plt.close()


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    obs_log = read_observation_log(args.observation_log_file[0])

    start_time = Time("2016-03-14T00:00:00.000")
    end_time = start_time + (52 * u.week)

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        print(f"Marking centers for filter {filter_to_string(filter_type)} ...")
        ml = match_within_timeframe(
            obs_log=obs_log, start_time=start_time, end_time=end_time
        )
        filter_mask = ml["FILTER"] == filter_type
        ml = ml[filter_mask]

        mark_comet_centers(swift_data, ml)


if __name__ == "__main__":
    sys.exit(main())
