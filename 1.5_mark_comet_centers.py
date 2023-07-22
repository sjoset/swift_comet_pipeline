#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log

import matplotlib.pyplot as plt

from tqdm import tqdm

from astropy.visualization import ZScaleInterval
from astropy.io import fits

from argparse import ArgumentParser

from configs import read_swift_project_config
from swift_types import (
    SwiftData,
    SwiftObservationLog,
    SwiftFilter,
    filter_to_file_string,
)
from observation_log import (
    # match_by_orbit_ids_and_filters,
    read_observation_log,
    # match_within_timeframe,
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
        "swift_project_config", nargs=1, help="Filename of project config"
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


def mark_comet_centers(
    swift_data: SwiftData, obs_log: SwiftObservationLog, image_save_dir: pathlib.Path
) -> None:
    """
    Takes an observation log, finds images in uvv and uw1 filters, and outputs pngs images of each
    observation annotated with the center of the comet marked.
    Output images are placed in image_save_dir/[filter]/
    """
    plt.rcParams["figure.figsize"] = (15, 15)

    # directories to store the uw1 and uvv images: image_save_dir/[filter]/
    dir_by_filter = {
        SwiftFilter.uw1: image_save_dir
        / pathlib.Path(filter_to_file_string(SwiftFilter.uw1)),
        SwiftFilter.uvv: image_save_dir
        / pathlib.Path(filter_to_file_string(SwiftFilter.uvv)),
    }
    # create directories we will need if they don't exist
    for fdir in dir_by_filter.values():
        fdir.mkdir(parents=True, exist_ok=True)

    # num_to_process = len(obs_log.index)
    # num_processed = 0

    progress_bar = tqdm(obs_log.iterrows(), total=obs_log.shape[0])
    # for every entry in the observation log,
    for _, row in progress_bar:
        # print(
        #     f"Processing image {num_processed}/{num_to_process} ({num_processed*100.0/num_to_process:03.1f}%) ...\r"
        # )
        obsid = row["OBS_ID"]
        extension = row["EXTENSION"]
        px = round(float(row["PX"]))
        py = round(float(row["PY"]))
        filter_str = filter_to_file_string(row["FILTER"])  # type: ignore

        # ask where the raw swift data FITS file is and read it
        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        output_image_name = pathlib.Path(f"{obsid}_{extension}_{filter_str}.png")
        output_image_path = dir_by_filter[row["FILTER"]] / output_image_name  # type: ignore

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_data)

        im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)
        fig.colorbar(im1)
        # mark comet center
        plt.axvline(px, color="w", alpha=0.3)
        plt.axhline(py, color="w", alpha=0.3)

        ax1.set_title("C/2013US10")
        ax1.set_xlabel(f"{row['MID_TIME']}")
        ax1.set_ylabel(f"{row['FITS_FILENAME']}")

        plt.savefig(output_image_path)
        plt.close()
        # num_processed += 1

        progress_bar.set_description(f"Processed {obsid} extension {extension}")

    print("")


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None or swift_project_config.observation_log is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_project_config.swift_data_path)
        .expanduser()
        .resolve()
    )

    obs_log = read_observation_log(swift_project_config.observation_log)

    image_save_dir = (
        swift_project_config.product_save_path.expanduser().resolve()
        / pathlib.Path("centers")
    )

    # we only care about uw1 and uvv
    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        print(f"Marking centers for filter {filter_to_file_string(filter_type)} ...")
        # ml = match_within_timeframe(
        #     obs_log=obs_log, start_time=start_time, end_time=end_time
        # )
        filter_mask = obs_log["FILTER"] == filter_type
        ml = obs_log[filter_mask]
        # filter_mask = ml["FILTER"] == filter_type
        # ml = ml[filter_mask]

        mark_comet_centers(
            swift_data=swift_data, obs_log=ml, image_save_dir=image_save_dir
        )


if __name__ == "__main__":
    sys.exit(main())
