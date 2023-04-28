#!/usr/bin/env python3

import os
import pathlib
import sys
import pandas as pd
import logging as log
import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)

from argparse import ArgumentParser

from swift_types import SwiftFilter, SwiftObservationID
from read_swift_config import read_swift_config
from swift_data import SwiftData
from stacking import stack_by_obsid, write_stacked_image


__version__ = "0.0.1"


def show_fits_scaled(image_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)

    im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax, origin="lower")
    fig.colorbar(im1)

    plt.axvline(image_data.shape[1] / 2, color="b", alpha=0.1)
    plt.axhline(image_data.shape[0] / 2, color="b", alpha=0.1)

    plt.show()


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
        "observation_log_file", nargs=1, help="Filename of observation log input"
    )
    parser.add_argument("obsid", nargs=1, help="Observation id to stack")

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

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )
    obs_log = pd.read_csv(args.observation_log_file[0])
    stacked_image_dir = pathlib.Path(swift_config["stacked_image_dir"])

    obsid = SwiftObservationID(args.obsid[0])

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        stacked_image = stack_by_obsid(
            swift_data=swift_data,
            obs_log=obs_log,
            obsid=obsid,
            filter_type=filter_type,
        )
        if stacked_image is None:
            print(
                f"Stacking unsuccessful for observation id {obsid} in filter {str(filter_type)} :("
            )
            continue

        print(f"Total exposure time: {stacked_image.exposure_time}")
        print("Contributing source images:")
        for i in stacked_image.sources:
            print(f"Obsid: {i[0]}, filename: {i[1]}, extension: {i[2]}")

        show_fits_scaled(stacked_image.stacked_image)

        write_stacked_image(stacked_image_dir, stacked_image)


if __name__ == "__main__":
    sys.exit(main())
