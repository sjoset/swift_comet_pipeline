#!/usr/bin/env python3

import os
import pathlib
import sys
import itertools
import logging as log
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time

from astropy.visualization import (
    ZScaleInterval,
)

from argparse import ArgumentParser

from swift_types import (
    SwiftData,
    SwiftFilter,
    SwiftOrbitID,
    SwiftObservationLog,
    SwiftStackingMethod,
    SwiftPixelResolution,
    filter_to_string,
)
from read_swift_config import read_swift_config
from stacking import (
    stack_image_by_selection,
    write_stacked_image,
    includes_uvv_and_uw1_filters,
)
from swift_observation_log import read_observation_log, match_within_timeframe


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
    parser.add_argument("orbit", nargs=1, help="orbit id to stack")

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def do_stack(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    stacked_image_dir: pathlib.Path,
    do_coincidence_correction: bool,
    detector_scale: SwiftPixelResolution,
) -> None:
    # test if there are uvv and uw1 images in the data set
    (stackable, _) = includes_uvv_and_uw1_filters(obs_log=obs_log)
    if not stackable:
        print("The data does not have data in both filters!")

    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [SwiftStackingMethod.summation, SwiftStackingMethod.median]

    # TODO: build a dictionary of [uw1, uvv] filenames for [sum, median]: both images and info files
    #  then we can point the stacking loader in oh.py to this file so it can grab all of the images in both filters

    for filter_type, stacking_method in itertools.product(
        filter_types, stacking_methods
    ):
        print(
            f"Stacking for filter {filter_to_string(filter_type)}: stacking type {stacking_method} ..."
        )

        # now narrow down the data to just one filter at a time
        filter_mask = obs_log["FILTER"] == filter_type
        ml = obs_log[filter_mask]

        stacked_image = stack_image_by_selection(
            swift_data=swift_data,
            obs_log=ml,
            do_coincidence_correction=do_coincidence_correction,
            detector_scale=detector_scale,
            stacking_method=stacking_method,
        )

        if stacked_image is None:
            print("Stacking image failed :( ")
            continue

        print(stacked_image)

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

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )
    # obs_log = pd.read_csv(args.observation_log_file[0])
    obs_log = read_observation_log(args.observation_log_file[0])
    stacked_image_dir = pathlib.Path(swift_config["stacked_image_dir"])

    orbit_id = SwiftOrbitID(args.orbit[0])

    # start_time = Time("2016-03-14T00:00:00.000")
    # # end_time = start_time + (52 * u.week)
    # end_time = start_time + (8 * u.week)

    # time_match = match_within_timeframe(
    #     obs_log=obs_log, start_time=start_time, end_time=end_time
    # )

    obsid_match = obs_log[obs_log["ORBIT_ID"] == orbit_id]

    do_stack(
        swift_data=swift_data,
        obs_log=obsid_match,
        stacked_image_dir=stacked_image_dir,
        do_coincidence_correction=False,
        detector_scale=SwiftPixelResolution.data_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
