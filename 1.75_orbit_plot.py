#!/usr/bin/env python3

import os
import sys
import calendar
import logging as log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u

from bisect import bisect_left, bisect_right

# from datetime import datetime

from astropy.time import Time

from argparse import ArgumentParser

from swift_types import SwiftFilter, SwiftObservationLog

# from read_swift_config import read_swift_config
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


def make_orbit_and_observation_plots(comet_orbit_df, earth_orbit_df, obs_log):
    # add column of observation mid times as datetime objects
    obs_log["obs_datetime_mid"] = list(
        map(lambda x: x.to_datetime(), obs_log["MID_TIME"].values)
    )
    obs_log.index = obs_log["obs_datetime_mid"]

    # text labels
    label_xs, label_ys, labels = comet_observation_labels_by_month(
        obs_log=obs_log, comet_orbit_df=comet_orbit_df
    )
    label_rs = np.sqrt(label_xs**2 + label_ys**2)
    label_thetas = np.arctan2(label_ys, label_xs)

    # observation points
    observation_rs, observation_thetas = find_observation_coords_from_obs_log(
        obs_log=obs_log, comet_orbit_df=comet_orbit_df
    )

    # comet orbit curve
    comet_rs = np.sqrt(comet_orbit_df["x"] ** 2 + comet_orbit_df["y"] ** 2)
    comet_thetas = np.arctan2(comet_orbit_df["y"], comet_orbit_df["x"])

    # earth orbit curve
    earth_rs = np.sqrt(earth_orbit_df["x"] ** 2 + earth_orbit_df["y"] ** 2)
    earth_thetas = np.arctan2(earth_orbit_df["y"], earth_orbit_df["x"])

    # plt.rcParams["figure.figsize"] = (15, 15)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="polar")

    # ax.plot(comet_thetas, comet_rs, lw=1.5, alpha=0.2, color="#688894")
    # ax.plot(earth_thetas, earth_rs, lw=1.0, alpha=0.2, color="#afac7c")
    # ax.scatter(observation_thetas, observation_rs, color="#a4b7be", alpha=0.5, zorder=1)

    # for r, theta, label in zip(label_rs, label_thetas, labels):
    #     text_rot = theta * (180 / np.pi)
    #     if text_rot > 90 and text_rot < 180:
    #         text_rot -= 180
    #     elif text_rot > 180:
    #         text_rot -= 360
    #     plt.text(
    #         theta,
    #         r,
    #         label,
    #         rotation=text_rot,
    #         color="#688894",
    #         ha="center",
    #         va="center",
    #         alpha=1.0,
    #     )
    #
    # ax.set_title("C/2013US10")

    # ax.set_yticklabels([])
    # ax.axis("off")

    # fig.savefig("all.pdf")

    # plt.show()
    # plt.close()

    generate_orbits_figure(
        comet_rs,
        comet_thetas,
        earth_rs,
        earth_thetas,
        observation_rs,
        observation_thetas,
        "orbits.pdf",
    )
    generate_labels_figure(
        comet_rs,
        comet_thetas,
        earth_rs,
        earth_thetas,
        observation_rs,
        observation_thetas,
        label_rs,
        label_thetas,
        labels,
        "observation_labels.pdf",
    )
    generate_combined_figure(
        comet_rs,
        comet_thetas,
        earth_rs,
        earth_thetas,
        observation_rs,
        observation_thetas,
        label_rs,
        label_thetas,
        labels,
        out_file="orbits_with_labels_combined.pdf",
    )


def generate_orbits_figure(
    comet_rs,
    comet_thetas,
    earth_rs,
    earth_thetas,
    observation_rs,
    observation_thetas,
    out_file=None,
):
    plt.rcParams["figure.figsize"] = (15, 15)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_yticklabels([])
    ax.axis("off")
    draw_earth_orbit_plot(earth_rs, earth_thetas, ax, alpha=0.2)
    draw_comet_orbit_plot(comet_rs, comet_thetas, ax, alpha=0.2)
    draw_observations_plot(observation_rs, observation_thetas, ax, alpha=1.0)

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)

    plt.close()


def generate_labels_figure(
    # observation_rs, observation_thetas, label_rs, label_thetas, labels, out_file=None
    comet_rs,
    comet_thetas,
    earth_rs,
    earth_thetas,
    observation_rs,
    observation_thetas,
    label_rs,
    label_thetas,
    labels,
    out_file=None,
):
    # plt.rcParams["figure.figsize"] = (15, 15)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="polar")
    # ax.set_yticklabels([])
    # ax.axis("off")
    # draw_observation_labels_plot(
    #     observation_rs, observation_thetas, label_rs, label_thetas, labels, ax, plt
    # )
    #
    # if out_file is None:
    #     plt.show()
    # else:
    #     plt.savefig(out_file)
    #
    # plt.close()

    plt.rcParams["figure.figsize"] = (15, 15)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_yticklabels([])
    ax.axis("off")
    draw_earth_orbit_plot(earth_rs, earth_thetas, ax, alpha=0)
    draw_comet_orbit_plot(comet_rs, comet_thetas, ax, alpha=0)
    draw_observations_plot(observation_rs, observation_thetas, ax, alpha=0)
    draw_observation_labels_plot(
        observation_rs,
        observation_thetas,
        label_rs,
        label_thetas,
        labels,
        ax,
        plt,
        alpha=1.0,
    )

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)

    plt.close()


def generate_combined_figure(
    comet_rs,
    comet_thetas,
    earth_rs,
    earth_thetas,
    observation_rs,
    observation_thetas,
    label_rs,
    label_thetas,
    labels,
    out_file=None,
):
    plt.rcParams["figure.figsize"] = (15, 15)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_yticklabels([])
    ax.axis("off")
    draw_earth_orbit_plot(earth_rs, earth_thetas, ax, alpha=0.2)
    draw_comet_orbit_plot(comet_rs, comet_thetas, ax, alpha=0.2)
    draw_observations_plot(observation_rs, observation_thetas, ax, alpha=1.0)
    draw_observation_labels_plot(
        observation_rs,
        observation_thetas,
        label_rs,
        label_thetas,
        labels,
        ax,
        plt,
        alpha=1.0,
    )

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)

    plt.close()


def draw_earth_orbit_plot(earth_rs, earth_thetas, ax, alpha):
    ax.plot(earth_thetas, earth_rs, lw=1.0, alpha=alpha, color="#afac7c")


def draw_comet_orbit_plot(comet_rs, comet_thetas, ax, alpha):
    ax.plot(comet_thetas, comet_rs, lw=1.5, alpha=alpha, color="#688894")


def draw_observations_plot(observation_rs, observation_thetas, ax, alpha):
    ax.scatter(observation_thetas, observation_rs, color="#a4b7be", alpha=alpha)


def draw_observation_labels_plot(
    observation_rs, observation_thetas, label_rs, label_thetas, labels, ax, p, alpha
):
    ax.scatter(observation_thetas, observation_rs, color="#a4b7be", alpha=0, zorder=1)
    for r, theta, label in zip(label_rs, label_thetas, labels):
        text_rot = theta * (180 / np.pi)
        if text_rot > 90 and text_rot < 180:
            text_rot -= 180
        elif text_rot > 180:
            text_rot -= 360
        p.text(
            theta,
            r,
            label,
            rotation=text_rot,
            color="#688894",
            ha="center",
            va="center",
            alpha=alpha,
        )


# def orbit_plot(comet_df, earth_df, observation_rs, observation_thetas):
#     c_rs = np.sqrt(comet_df["x"] ** 2 + comet_df["y"] ** 2)
#     c_thetas = np.arctan2(comet_df["y"], comet_df["x"])
#
#     e_rs = np.sqrt(earth_df["x"] ** 2 + earth_df["y"] ** 2)
#     e_thetas = np.arctan2(earth_df["y"], earth_df["x"])
#
#     plt.rcParams["figure.figsize"] = (15, 15)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection="polar")
#
#     # ax.vlines(times_at_observations, 0, )
#     ax.plot(c_thetas, c_rs, lw=1.5, alpha=0.5, color="#688894")
#     ax.plot(e_thetas, e_rs, lw=1.0, alpha=0.5, color="#afac7c")
#     ax.scatter(observation_thetas, observation_rs, color="#a4b7be", alpha=1.0, zorder=1)
#
#     ax.set_title("C/2013US10")
#     ax.set_yticklabels([])
#     ax.axis("off")
#
#     plt.show()
#     plt.close()


def get_closest_index_to_value(df, value, column):
    # given a pandas dataframe, look in 'column' for 'value'
    # return the index of the value if it is found, or the index of the closest in column
    lower_idx = bisect_left(df[column].values, value)
    higher_idx = bisect_right(df[column].values, value)

    if higher_idx != lower_idx:
        return lower_idx
    else:
        lower_idx = lower_idx - 1
        higher_diff = np.abs(df.iloc[higher_idx][column] - value)
        lower_diff = np.abs(df.iloc[lower_idx][column] - value)
        if lower_diff < higher_diff:
            return lower_idx
        else:
            return higher_idx


def find_observation_coords_from_obs_log(
    obs_log: SwiftObservationLog, comet_orbit_df: pd.DataFrame
):
    # pick out the observations in uw1 or uvv filters
    uw1_mask = obs_log["FILTER"] == SwiftFilter.uw1
    uvv_mask = obs_log["FILTER"] == SwiftFilter.uvv
    obs_log = obs_log[uw1_mask | uvv_mask]

    xs = []
    ys = []
    # find closest time to MID_TIME in orbit data and take those coords
    for _, row in obs_log.iterrows():
        t_match = row["MID_TIME"]
        df_idx_match = get_closest_index_to_value(comet_orbit_df, t_match, "jd")

        xs.append(comet_orbit_df.iloc[df_idx_match]["x"])
        ys.append(comet_orbit_df.iloc[df_idx_match]["y"])

    xs = np.array(xs)
    ys = np.array(ys)
    rs = np.sqrt(xs**2 + ys**2)
    thetas = np.arctan2(ys, xs)

    return rs, thetas


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


def comet_observation_labels_by_month(
    obs_log: SwiftObservationLog, comet_orbit_df: pd.DataFrame
):
    xs = []
    ys = []
    labels = []

    for gname, group in obs_log.groupby(pd.Grouper(freq="1M")):
        if len(group) == 0:
            continue
        t_match = Time(str(np.mean(group["obs_datetime_mid"])), format="iso")

        t_idx = get_closest_index_to_value(comet_orbit_df, t_match, "jd")

        xs.append(comet_orbit_df.iloc[t_idx]["x"])
        ys.append(comet_orbit_df.iloc[t_idx]["y"])
        label_time = Time(gname).ymdhms
        labels.append(f"{calendar.month_abbr[label_time.month]} {label_time.year}")

    return np.array(xs), np.array(ys), labels


# def orbit_label_figure(xs, ys, labels):
#     rs = np.sqrt(xs**2 + ys**2)
#     thetas = np.arctan2(ys, xs)
#
#     plt.rcParams["figure.figsize"] = (15, 15)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection="polar")
#
#     ax.scatter(thetas, rs, color="#a4b7be", alpha=0.01, zorder=1)
#
#     for r, theta, label in zip(rs, thetas, labels):
#         text_rot = theta * (180 / np.pi)
#         if text_rot > 90 and text_rot < 180:
#             text_rot -= 180
#         elif text_rot > 180:
#             text_rot -= 360
#         plt.text(
#             theta,
#             r,
#             label,
#             rotation=text_rot,
#             color="#688894",
#             ha="center",
#             va="center",
#             alpha=0.5,
#         )
#
#     ax.set_title("C/2013US10")
#     ax.set_yticklabels([])
#     ax.axis("off")
#
#     plt.show()
#     plt.close()


def main():
    args = process_args()

    obs_log = read_observation_log(args.observation_log_file[0])

    comet_orbit = pd.read_csv("data/comet_orbital_data_horizons.csv")
    comet_orbit["jd"] = list(
        map(lambda t: Time(t, format="jd"), comet_orbit["datetime_jd"])
    )

    earth_orbit = pd.read_csv("data/earth_orbital_data_horizons.csv")

    # rs, thetas = find_observation_coords_from_obs_log(obs_log, comet_orbit)

    # orbit_plot(comet_orbit, earth_orbit, observation_rs=rs, observation_thetas=thetas)

    # obs_log["obs_datetime_mid"] = list(
    #     map(lambda x: x.to_datetime(), obs_log["MID_TIME"].values)
    # )
    # obs_log.index = obs_log["obs_datetime_mid"]
    # xs, ys, labels = orbit_labels(obs_log, comet_orbit)
    # orbit_label_figure(xs, ys, labels)

    make_orbit_and_observation_plots(
        obs_log=obs_log, comet_orbit_df=comet_orbit, earth_orbit_df=earth_orbit
    )


if __name__ == "__main__":
    sys.exit(main())
