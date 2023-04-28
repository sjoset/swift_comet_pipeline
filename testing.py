#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits
from argparse import ArgumentParser
from typing import List

from read_swift_config import read_swift_config
from swift_data import SwiftData, swift_orbit_id_from_obsid
from swift_types import (
    SwiftObservationLog,
    SwiftObservationID,
    SwiftFilter,
    SwiftOrbitID,
    PixelCoord,
    swift_observation_id_from_int,
)
from swift_observation_log import (
    match_by_obsid_and_filter,
    match_by_orbit_id_and_filter,
)
from stacking import (
    # get_image_dimensions_to_center_comet,
    determine_stacking_image_size,
    center_image_on_coords,
    # coincidence_loss_correction,
    # stack_by_obsid,
)
import sushi


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


# TODO: a similar function could identify
def test_observation_log_obsid_match(obs_log: SwiftObservationLog) -> None:
    for obsid_int in np.unique(obs_log["OBS_ID"]):
        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            obsid = swift_observation_id_from_int(obsid_int)
            if obsid is None:
                continue

            newdf = match_by_obsid_and_filter(
                obs_log,
                obsid,
                filter_type=filter_type,
            )

            for _, row in newdf.iterrows():
                print(f"{row['OBS_ID']} | {row['EXTENSION']} | {row['FILTER']}")


def test_observation_log_orbit_match(obs_log: SwiftObservationLog) -> None:
    obsid_ints = np.unique(obs_log["OBS_ID"])
    obsids = map(SwiftObservationID, obsid_ints)
    orbit_ids = np.unique(list(map(swift_orbit_id_from_obsid, obsids)))
    # print(obsid_ints, obsids, orbit_ids)

    for orbit in orbit_ids:
        contains_filters = {SwiftFilter.uvv: False, SwiftFilter.uw1: False}
        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            newdf = match_by_orbit_id_and_filter(
                obs_log,
                orbit,
                filter_type=filter_type,
            )
            if len(newdf) != 0:
                contains_filters[filter_type] = True

            # for _, row in newdf.iterrows():
            #     print(f"{row['OBS_ID']} | {row['EXTENSION']} | {row['FILTER']}")

        if all(filter_found == True for filter_found in contains_filters.values()):
            print(f"Candidate for OH subtraction: orbit {orbit.orbit_id}")
        # else:
        #     print(f"Not a candidate for OH subtraction: orbit {orbit.orbit_id}")


def show_fits_scaled(image_data_one, image_data_two, image_data_three):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data_one)

    im1 = ax1.imshow(image_data_one, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(image_data_two, vmin=vmin, vmax=vmax)
    im3 = ax3.imshow(image_data_three, vmin=vmin, vmax=vmax)
    # fig.colorbar(im1)
    # fig.colorbar(im2)
    fig.colorbar(im3)

    plt.show()


def test_image_resizing_method_vs_lucy(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    obsids: List[SwiftObservationID],
    filter_type: SwiftFilter,
) -> None:
    # Look at our resized images vs a poorly chosen a-priori guessed size

    for obsid in obsids:
        matching_obs_log = match_by_obsid_and_filter(
            obs_log=obs_log, obsid=obsid, filter_type=filter_type
        )
        for _, row in matching_obs_log.iterrows():
            # Get list of image files for this filter and image type
            img_file_list = swift_data.get_swift_uvot_image_paths(
                obsid=obsid, filter_type=filter_type
            )
            if img_file_list is None:
                print(f"No files found for {obsid=} and {filter_type=}")
                continue

            stacking_image_size = determine_stacking_image_size(
                swift_data=swift_data,
                obs_log=obs_log,
                obsid=obsid,
                filter_type=filter_type,
            )

            for img_file in img_file_list:
                print(
                    f"Processing {img_file.name}, extension {row['EXTENSION']}: Comet center at {row['PX']}, {row['PY']}"
                )

                image_data = fits.getdata(img_file, ext=row["EXTENSION"])

                # numpy uses (row, col) indexing - so (y, x)
                new_image = center_image_on_coords(
                    image_data,  # type: ignore
                    PixelCoord(x=row["PX"], y=row["PY"]),  # type: ignore
                    stacking_image_size=stacking_image_size,  # type: ignore
                )

                new_image2 = sushi.set_coord(
                    image_data,
                    np.array([row["PY"] - 1, row["PX"] - 1]),
                    (1500, 1500),
                )

                # show_fits_scaled(image_data, new_image)
                show_fits_scaled(image_data, new_image, new_image2)
                print("-------")


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    swift_data = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    # test_swift_data(swift_data)

    obs_log = pd.read_csv(args.observation_log_file[0])
    # test_observation_log_obsid_match(obs_log)
    test_observation_log_orbit_match(obs_log)

    # obsid_strings = ["00020405001", "00020405002", "00020405003", "00034318005"]
    # # obsid_strings = [
    # #     "00034316001",
    # #     "00034316002",
    # #     "00034318001",
    # #     "00034318002",
    # #     "00034318003",
    # # ]
    # obsids = list(map(lambda x: SwiftObservationID(x), obsid_strings))
    # test_image_resizing_method_vs_lucy(
    #     swift_data=swift_data,
    #     obs_log=obs_log,
    #     obsids=obsids,
    #     filter_type=SwiftFilter.uw1,
    # )


if __name__ == "__main__":
    sys.exit(main())
