#!/usr/bin/env python3

import os
import pathlib
import sys
import calendar
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time

from argparse import ArgumentParser

from swift_types import (
    SwiftFilter,
    SwiftObservationLog,
)
from read_swift_config import read_swift_config
from swift_observation_log import read_observation_log


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


def show_observation_timeline(orbit_labels, times_at_observations) -> None:
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

    ax.set_title("C/2013US10")
    # ax1.set_xlabel(f"{row['MID_TIME']}")
    # ax1.set_ylabel(f"{row['FITS_FILENAME']}")

    plt.show()
    plt.close()


def monthly_group_breakdown(group_time, df: SwiftObservationLog):
    gdate = Time(group_time).ymdhms

    print(f"{calendar.month_abbr[gdate.month]} {gdate.year}:")

    uw1_mask = df["FILTER"] == SwiftFilter.uw1
    uvv_mask = df["FILTER"] == SwiftFilter.uvv

    tstart, tend = np.min(df["MID_TIME"]), np.max(df["MID_TIME"])
    print(f"[ {tstart.to_datetime()} ] through [ {tend.to_datetime()} ]")
    print(
        f"\t{(tend - tstart).to_value(u.hr):03.1f} hours between first to last observation"
    )

    stackable_orbits = []
    stackable_times = []

    orbit_ids = np.unique(df["ORBIT_ID"])
    for orbit_id in orbit_ids:
        orbit_mask = df["ORBIT_ID"] == orbit_id
        uw1s_in_this_orbit = df[orbit_mask & uw1_mask]
        uvvs_in_this_orbit = df[orbit_mask & uvv_mask]
        num_uw1s = len(uw1s_in_this_orbit)
        num_uvvs = len(uvvs_in_this_orbit)
        print(f"orbit {orbit_id}: {num_uw1s} uw1 images, {num_uvvs} uvv images")
        if num_uw1s > 0 or num_uvvs > 0:
            # look through all the individual observations in this orbit
            for _, row in df[orbit_mask].iterrows():
                stackable_orbits.append(orbit_id)
                stackable_times.append(row["MID_TIME"])

    print(f"Stack from orbit {np.min(orbit_ids)} to {np.max(orbit_ids)}")

    print("")

    return stackable_orbits, stackable_times


def main():
    args = process_args()

    obs_log = read_observation_log(args.observation_log_file[0])

    obs_log["obs_datetime_mid"] = list(
        map(lambda x: x.to_datetime(), obs_log["MID_TIME"].values)
    )
    obs_log.index = obs_log["obs_datetime_mid"]
    # obs_log.groupby(pd.Grouper(freq="1M"))

    total_orbits = []
    total_times = []

    for gname, group in obs_log.groupby(pd.Grouper(freq="1M")):
        if len(group) == 0:
            continue
        orbs, ts = monthly_group_breakdown(gname, group)
        total_orbits.append(orbs)
        total_times.append(ts)

    # flatten lists
    total_orbits = [item for sublist in total_orbits for item in sublist]
    total_times = [item for sublist in total_times for item in sublist]

    show_observation_timeline(total_orbits, total_times)


if __name__ == "__main__":
    sys.exit(main())
