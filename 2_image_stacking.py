#!/usr/bin/env python3

import os
import pathlib
import sys
import itertools
import logging as log
import matplotlib.pyplot as plt

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
from swift_observation_log import read_observation_log
from stack_info import stackinfo_from_stacked_images


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
        "--stackinfo",
        "-s",
        default=None,
        help="filename for saving stacking information",
    )
    parser.add_argument(
        "observation_log_file", nargs=1, help="Filename of observation log input"
    )
    parser.add_argument(
        "stacked_image_dir", nargs=1, help="directory to store stacked images and info"
    )
    parser.add_argument("orbit_start", nargs=1, help="first orbit id to stack")
    parser.add_argument("orbit_end", nargs=1, help="last orbit id to stack")

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
    stackinfo_output_path: pathlib.Path,
    do_coincidence_correction: bool,
    detector_scale: SwiftPixelResolution,
) -> None:
    # test if there are uvv and uw1 images in the data set
    (stackable, _) = includes_uvv_and_uw1_filters(obs_log=obs_log)
    if not stackable:
        print("The data does not have data in both filters!")

    # Do both filters with sum and median stacking
    filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
    stacking_methods = [SwiftStackingMethod.summation, SwiftStackingMethod.median]

    stacking_outputs = {}

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
            print("Stacking image failed, skipping... ")
            continue

        stacked_path, stacked_info_path = write_stacked_image(
            stacked_image_dir=stacked_image_dir, stacked_image=stacked_image
        )
        print(f"Wrote to {stacked_path}, info at {stacked_info_path}")

        stacking_outputs[(filter_type, stacking_method)] = (
            str(stacked_path),
            str(stacked_info_path),
        )

    stackinfo_from_stacked_images(stackinfo_output_path, stacking_outputs)


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    stacked_image_dir = pathlib.Path(args.stacked_image_dir[0])

    if args.stackinfo is None:
        stackinfo_output_path = stacked_image_dir / pathlib.Path("stack.json")
    else:
        stackinfo_output_path = stacked_image_dir / pathlib.Path(args.stackinfo)

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    obs_log = read_observation_log(args.observation_log_file[0])

    # orbit_id = SwiftOrbitID(args.orbit[0])
    # orbit_match = obs_log[obs_log["ORBIT_ID"] == orbit_id]
    # do_stack(
    #     swift_data=swift_data,
    #     obs_log=orbit_match,
    #     stacked_image_dir=stacked_image_dir,
    #     do_coincidence_correction=False,
    #     detector_scale=SwiftPixelResolution.data_mode,
    # )

    # start_time = Time("2016-03-14T00:00:00.000")
    # # end_time = start_time + (52 * u.week)
    # end_time = start_time + (8 * u.week)

    # time_match = match_within_timeframe(
    #     obs_log=obs_log, start_time=start_time, end_time=end_time
    # )

    orbit_start = int(SwiftOrbitID(args.orbit_start[0]))
    orbit_end = int(SwiftOrbitID(args.orbit_end[0]))
    obs_log["orbit_ints"] = obs_log["ORBIT_ID"].map(int)

    mask_start = obs_log["orbit_ints"] >= orbit_start
    mask_end = obs_log["orbit_ints"] <= orbit_end

    # orbit_match = obs_log[
    #     obs_log["orbit_ints"] >= orbit_start & obs_log["orbit_ints"] <= orbit_end
    # ]
    orbit_match = obs_log[mask_start & mask_end]

    do_stack(
        swift_data=swift_data,
        obs_log=orbit_match,
        stacked_image_dir=stacked_image_dir,
        stackinfo_output_path=stackinfo_output_path,
        do_coincidence_correction=True,
        detector_scale=SwiftPixelResolution.event_mode,
    )


if __name__ == "__main__":
    sys.exit(main())
