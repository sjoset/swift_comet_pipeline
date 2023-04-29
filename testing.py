#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits
from astropy.time import Time
from argparse import ArgumentParser

from read_swift_config import read_swift_config
from swift_data import SwiftData
from swift_types import (
    SwiftObservationLog,
    SwiftObservationID,
    SwiftFilter,
    SwiftOrbitID,
    PixelCoord,
)
from swift_observation_log import (
    match_by_obsids_and_filters,
    # match_by_orbit_ids_and_filters,
    read_observation_log,
    get_obsids_in_orbits,
    match_within_timeframe,
)
from stacking import (
    determine_stacking_image_size,
    center_image_on_coords,
    is_OH_stackable,
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


def test_observation_log_obsid_match(obs_log: SwiftObservationLog) -> None:
    # for every observation ID, match uw1 and uvv filters and print the results
    for obsid in np.unique(obs_log["OBS_ID"]):
        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            newdf = match_by_obsids_and_filters(
                obs_log,
                obsids=[obsid],
                filter_types=[filter_type],
            )

            for _, row in newdf.iterrows():
                print(f"{row['OBS_ID']} | {row['EXTENSION']} | {row['FILTER']}")


def test_observation_log_multi_match(obs_log: SwiftObservationLog) -> None:
    obsids = list(map(SwiftObservationID, ["00034423001", "00033824004"]))
    filters = [SwiftFilter.uw1, SwiftFilter.ugrism]

    matching_obs_log = match_by_obsids_and_filters(
        obs_log=obs_log, obsids=obsids, filter_types=filters
    )
    print(matching_obs_log)
    print(matching_obs_log["FILTER"], matching_obs_log["OBS_ID"])


def test_OH_stackable(obs_log: SwiftObservationLog) -> None:
    # orbit_ids = list(map(SwiftOrbitID, ["00033759", "00033760"]))
    orbit_ids = list(
        map(SwiftOrbitID, ["00033759", "00033760", "00033822", "00033826", "00033827"])
    )
    a = is_OH_stackable(obs_log=obs_log, orbit_ids=orbit_ids)
    print(a)

    # has_both = set({})
    # for orbit in orbit_ids:
    #     contains_filters = {SwiftFilter.uvv: False, SwiftFilter.uw1: False}
    #     for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
    #         newdf = match_by_orbit_ids_and_filters(
    #             obs_log,
    #             orbit_ids=[orbit],
    #             filter_types=[filter_type],
    #         )
    #         if len(newdf) != 0:
    #             contains_filters[filter_type] = True
    #
    #         # for _, row in newdf.iterrows():
    #         #     print(f"{row['OBS_ID']} | {row['EXTENSION']} | {row['FILTER']}")
    #
    #     if all(filter_found == True for filter_found in contains_filters.values()):
    #         # print(f"Candidate for OH subtraction: orbit {orbit}")
    #         has_both.add(orbit)
    # print(has_both)


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
) -> None:
    obsid_strings = ["00020405001", "00020405002", "00020405003", "00034318005"]
    # # obsid_strings = [
    # #     "00034316001",
    # #     "00034316002",
    # #     "00034318001",
    # #     "00034318002",
    # #     "00034318003",
    # # ]
    obsids = list(map(SwiftObservationID, obsid_strings))
    filter_type = SwiftFilter.uw1

    # Look at our resized images vs a poorly chosen a-priori guessed size
    for obsid in obsids:
        matching_obs_log = match_by_obsids_and_filters(
            obs_log=obs_log, obsids=[obsid], filter_types=[filter_type]
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


def test_obsids_from_orbits(obs_log: SwiftObservationLog) -> None:
    orbit_ids = list(
        map(SwiftOrbitID, ["00033759", "00033760", "00033822", "00033826", "00033827"])
    )
    a = get_obsids_in_orbits(obs_log, orbit_ids=orbit_ids)
    print(a)


def test_timeframe_matching(obs_log: SwiftObservationLog) -> None:
    start_time = Time("2014-12-19T00:27:21.000")
    end_time = Time("2015-09-01T12:04:21.000")
    ml = match_within_timeframe(
        obs_log=obs_log, start_time=start_time, end_time=end_time
    )

    # print(ml["MID_TIME"])
    ts = ml["MID_TIME"].values
    ts.sort()
    print(ts)


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

    obs_log = read_observation_log(args.observation_log_file[0])

    test_timeframe_matching(obs_log=obs_log)

    # test_observation_log_obsid_match(obs_log)

    # test_observation_log_multi_match(obs_log)

    # test_OH_stackable(obs_log)

    # test_image_resizing_method_vs_lucy(swift_data=swift_data, obs_log=obs_log)


if __name__ == "__main__":
    sys.exit(main())
