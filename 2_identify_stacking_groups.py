#!/usr/bin/env python3

import os
import copy
import pathlib
import sys
import calendar
import logging as log
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time

from argparse import ArgumentParser
from typing import List

from configs import read_swift_project_config
from swift_types import SwiftFilter, SwiftObservationLog, SwiftObservationID

from observation_log import read_observation_log
from user_input import get_selection, get_float, get_yes_no


__version__ = "0.0.1"


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


def show_observation_timeline(
    orbit_id_labels: List[SwiftObservationID], times_at_observations: List[Time]
) -> None:
    ts = list(map(lambda x: x.to_datetime(), times_at_observations))

    plt.rcParams["figure.figsize"] = (15, 15)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.vlines(times_at_observations, 0, )
    ax.plot(
        ts,
        np.zeros_like(ts),
        "-o",
        color="k",
        markerfacecolor="w",
    )

    # ax.set_title("C/2013US10")
    # ax1.set_xlabel(f"{row['MID_TIME']}")
    # ax1.set_ylabel(f"{row['FITS_FILENAME']}")

    plt.show()
    plt.close()


# def epochs_from_time_window(
#     obs_log: SwiftObservationLog, time_window: u.Quantity
# ) -> List[SwiftObservationLog]:
#     # sort by observations by time, oldest first
#     obs_log = obs_log.sort_values(by="MID_TIME", ascending=True).reset_index(drop=True)
#
#     epoch_list = []
#     epoch_count = 0
#
#     while True:
#         # get oldest observation date in first row (iloc[0]) and add time_window to it, group the data in this time window
#         t_start = Time(obs_log.iloc[0].MID_TIME) - 1 * u.s
#         t_end = t_start + time_window + 1 * u.s
#         t_start, t_end = t_start.to_datetime(), t_end.to_datetime()
#         time_filter = (obs_log.MID_TIME > t_start) & (obs_log.MID_TIME < t_end)
#
#         epoch_list.append(obs_log[time_filter].copy())
#         epoch_count += 1
#
#         old_cutoff_filter = obs_log.MID_TIME > t_end
#         obs_log = obs_log[old_cutoff_filter]
#
#         # check to see if there is any data left
#         if obs_log.empty:
#             break
#
#     return epoch_list


def epochs_from_time_delta(
    obs_log: SwiftObservationLog, max_time_delta: u.Quantity
) -> List[SwiftObservationLog]:
    # sort by observations by time, oldest first
    obs_log = obs_log.sort_values(by="MID_TIME", ascending=True).reset_index(drop=True)

    epoch_list = []
    epoch_count = 0

    while True:
        max_index = len(obs_log) - 1
        # print(f"{max_index=}")
        t_start = Time(obs_log.iloc[0].MID_TIME) - 1 * u.s

        # print(f"t_start: {t_start}")
        # keep checking if next observation is within max_time_delta
        prev_index = 0
        while True:
            prev_time = Time(obs_log.iloc[prev_index].MID_TIME)

            cur_index = prev_index + 1
            # print(f"{max_index=}, {prev_index=} {cur_index=}")

            cur_time = Time(obs_log.iloc[cur_index].MID_TIME)
            delta_t = cur_time - prev_time

            # is the time delta to the next observation too large?  Use the previous as the stopping point
            if delta_t > max_time_delta:
                t_end = prev_time + 1 * u.s
                break

            # is the current index the last row?  Use this last row as the stopping point
            if cur_index == max_index:
                t_end = cur_time + 1 * u.s
                break

            prev_index = cur_index

        # t_end = Time(obs_log.iloc[stop_index].MID_TIME) + 1 * u.s
        # print(f"t_end: {t_end}")
        t_start, t_end = t_start.to_datetime(), t_end.to_datetime()
        time_filter = (obs_log.MID_TIME > t_start) & (obs_log.MID_TIME < t_end)

        epoch_list.append(obs_log[time_filter].copy())
        epoch_count += 1

        old_cutoff_filter = obs_log.MID_TIME > t_end
        obs_log = obs_log[old_cutoff_filter]

        # check to see if there is any data left
        if obs_log.empty:
            break

    return epoch_list


# def timing_window_loop(obs_log: SwiftObservationLog) -> List[SwiftObservationLog]:
#     epoch_list = []
#
#     finished = False
#     time_units = [u.hour, u.day, u.s]
#
#     num_observations = len(obs_log)
#     print(f"Total number of uw1 or uvv observations in dataset: {num_observations}")
#
#     while finished is False:
#         time_window_raw = get_float("Enter time window for epoch: ")
#         time_unit = time_units[get_selection(time_units)]
#         time_window = time_window_raw * time_unit
#
#         epoch_list = epochs_from_time_window(obs_log=obs_log, time_window=time_window)
#         print(f"{len(epoch_list)} epochs identified with time window of {time_window}:")
#         for i, epoch in enumerate(epoch_list):
#             epoch_start_raw = Time(np.min(epoch.MID_TIME))
#             epoch_end_raw = Time(np.max(epoch.MID_TIME))
#             epoch_start = epoch_start_raw.ymdhms
#             epoch_end = epoch_end_raw.ymdhms
#             epoch_length = (epoch_end_raw - epoch_start_raw).to(u.hour)
#             observations_in_epoch = len(epoch)
#             uw1_observations = len(epoch[epoch.FILTER == SwiftFilter.uw1])
#             uvv_observations = len(epoch[epoch.FILTER == SwiftFilter.uvv])
#             if i == 0:
#                 prev_epoch = epoch_list[0]
#                 delta_t_str = ""
#             else:
#                 prev_epoch = epoch_list[i - 1]
#                 separation_from_prev_epoch = epoch_start_raw - Time(
#                     np.max(prev_epoch.MID_TIME)
#                 )
#                 delta_t_str = f"\tSeparation from last epoch: {separation_from_prev_epoch.to(u.day):05.1f} ({separation_from_prev_epoch.to(u.hour):07.1f})"
#             print(
#                 f"\tStart: {epoch_start.day:2d} {calendar.month_abbr[epoch_start.month]} {epoch_start.year}"
#                 + f"\tEnd: {epoch_end.day:2d} {calendar.month_abbr[epoch_end.month]} {epoch_end.year}"
#                 + f"\tDuration: {epoch_length:3.2f}"
#                 + f"\tObservations: {observations_in_epoch:3d}"
#                 + f"\tUW1: {uw1_observations:3d}"
#                 + f"\tUVV: {uvv_observations:3d}"
#                 + delta_t_str
#             )
#
#         total_in_epochs = sum([len(epoch) for epoch in epoch_list])
#         print(f"Total observations covered by epochs: {total_in_epochs}")
#
#         if total_in_epochs != num_observations:
#             print(
#                 f"There are {num_observations - total_in_epochs} observations excluded from epoch list!"
#                 + "Try adjusting the timing window."
#             )
#         else:
#             print("All observations accounted for in epoch selection!")
#
#         print("Try new time window?")
#         finished = not get_yes_no()
#
#     return epoch_list


def time_delta_loop(obs_log: SwiftObservationLog) -> List[SwiftObservationLog]:
    epoch_list = []

    finished = False
    time_units = [u.hour, u.day, u.s]

    num_observations = len(obs_log)
    print(f"Total number of uw1 or uvv observations in dataset: {num_observations}")

    while finished is False:
        time_delta_raw = get_float("Enter max time delta: ")
        time_unit = time_units[get_selection(time_units)]
        time_delta = time_delta_raw * time_unit

        epoch_list = epochs_from_time_delta(obs_log=obs_log, max_time_delta=time_delta)
        print(f"{len(epoch_list)} epochs identified with time delta of {time_delta}:")
        for i, epoch in enumerate(epoch_list):
            epoch_start_raw = Time(np.min(epoch.MID_TIME))
            epoch_end_raw = Time(np.max(epoch.MID_TIME))
            epoch_start = epoch_start_raw.ymdhms
            epoch_end = epoch_end_raw.ymdhms
            epoch_length = (epoch_end_raw - epoch_start_raw).to(u.hour)
            observations_in_epoch = len(epoch)
            uw1_observations = len(epoch[epoch.FILTER == SwiftFilter.uw1])
            uvv_observations = len(epoch[epoch.FILTER == SwiftFilter.uvv])
            if i == 0:
                prev_epoch = epoch_list[0]
                delta_t_str = ""
            else:
                prev_epoch = epoch_list[i - 1]
                separation_from_prev_epoch = epoch_start_raw - Time(
                    np.max(prev_epoch.MID_TIME)
                )
                delta_t_str = f"\tSeparation from last epoch: {separation_from_prev_epoch.to(u.day):05.1f} ({separation_from_prev_epoch.to(u.hour):07.1f})"
            print(
                f"\tStart: {epoch_start.day:2d} {calendar.month_abbr[epoch_start.month]} {epoch_start.year}"
                + f"\tEnd: {epoch_end.day:2d} {calendar.month_abbr[epoch_end.month]} {epoch_end.year}"
                + f"\tDuration: {epoch_length:3.2f}"
                + f"\tObservations: {observations_in_epoch:3d}"
                + f"\tUW1: {uw1_observations:3d}"
                + f"\tUVV: {uvv_observations:3d}"
                + delta_t_str
            )

        total_in_epochs = sum([len(epoch) for epoch in epoch_list])
        print(f"Total observations covered by epochs: {total_in_epochs}")

        if total_in_epochs != num_observations:
            print(
                f"There are {num_observations - total_in_epochs} observations excluded from epoch list!"
                + "Try adjusting the timing window."
            )
        else:
            print("All observations accounted for in epoch selection!")

        print("Try new time window?")
        finished = not get_yes_no()

    return epoch_list


def file_name_from_epoch(epoch: SwiftObservationLog) -> str:
    epoch_start = Time(np.min(epoch.MID_TIME)).ymdhms
    day = epoch_start.day
    month = calendar.month_abbr[epoch_start.month]
    year = epoch_start.year

    return f"{year}_{day:02d}_{month}"


# def vary_time_window(obs_log: SwiftObservationLog) -> None:
#     for time_window in range(24, 100, 1) * u.hour:  # type: ignore
#         epoch_list = epochs_from_time_window(
#             obs_log=obs_log.copy(), time_window=time_window
#         )
#         epoch_separations = []
#         for i, epoch in enumerate(epoch_list):
#             epoch_start = Time(np.min(epoch.MID_TIME))
#             if i == 0:
#                 prev_epoch = epoch_list[0]
#                 separation_from_prev_epoch = epoch_start - epoch_start
#             else:
#                 prev_epoch = epoch_list[i - 1]
#                 separation_from_prev_epoch = epoch_start - Time(
#                     np.max(prev_epoch.MID_TIME)
#                 )
#             epoch_separations.append(separation_from_prev_epoch.to_value(u.hour))
#         epoch_separations = epoch_separations[1:]
#         min_separation = np.min(epoch_separations)
#         # mean_separation = np.mean(epoch_separations)
#         sum_separation = np.sum(epoch_separations)
#         print(
#             f"Time window {time_window}\tMin: {min_separation} hours\tSum: {sum_separation}\tProduct: {min_separation * sum_separation}"
#         )


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None or swift_project_config.observation_log is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    obs_log = read_observation_log(swift_project_config.observation_log)
    # only uw1 and uvv filters
    filter_mask = (obs_log["FILTER"] == SwiftFilter.uw1) | (
        obs_log["FILTER"] == SwiftFilter.uvv
    )
    obs_log = obs_log[filter_mask]

    epoch_list = time_delta_loop(obs_log)
    path_list = [pathlib.Path(file_name_from_epoch(x)) for x in epoch_list]

    print("Save epochs?")
    save_epochs = get_yes_no()
    if save_epochs:
        epoch_dir = (
            swift_project_config.product_save_path.expanduser().resolve()
            / pathlib.Path("epochs")
        )
        epoch_dir.mkdir(exist_ok=True)
        print(f"Saving to {epoch_dir}")
        for i, (epoch, file_name) in enumerate(zip(epoch_list, path_list)):
            filename = f"{i:03d}_{file_name}.parquet"
            full_path = epoch_dir / filename
            print(f"File: {full_path}")
            epoch.to_parquet(full_path)


if __name__ == "__main__":
    sys.exit(main())
