import pandas as pd
import astropy.units as u
from astropy.time import Time

from swift_comet_pipeline.scp_types.primitive.swift_uvot_observation_log_dataframe import (
    SwiftUvotObservationLogDataframe,
)


def quantity_to_pandas_timedelta(t: u.Quantity) -> pd.Timedelta:

    if not t.unit.is_equivalent(u.s):  # type: ignore
        raise ValueError(f"{t} is not a time quantity!")

    return pd.to_timedelta(t.to_value(u.ns), unit="ns")  # type: ignore


def add_epoch_ids_by_time_window(
    obs_log: SwiftUvotObservationLogDataframe, max_time_between_obs: u.Quantity
):
    """
    Given a maximum time between observations, divide the observation log into epochs
    and assign each epoch a unique EpochID (a string based on the middle time of the epoch)
    """

    # copy and sort by observations by time
    df = obs_log.copy()
    df = df.sort_values(by="MID_TIME", ascending=True).reset_index(drop=True)

    # time format conversions
    max_gap = quantity_to_pandas_timedelta(t=max_time_between_obs)
    df["mid_time_temp"] = df.MID_TIME.dt.tz_localize("UTC")

    # calculate time difference between successive observations and assign epoch IDs
    # based on how many gaps we have had
    gap = df.mid_time_temp.diff()
    is_new_epoch = gap.gt(max_gap).fillna(True)
    df["epoch_id_int"] = is_new_epoch.cumsum()

    epoch_mean_times = (
        df.groupby("epoch_id_int", sort=False).mid_time_temp.mean().dt.round("min")
    )
    epoch_date_strings = epoch_mean_times.dt.strftime("%Y_%b_%d")

    # use three digits for epoch prefixes: 000_2020_03_01
    prefix_width = 3
    epoch_prefixes = (
        epoch_mean_times.index.astype(int)
        .to_series(index=epoch_mean_times.index)
        .map(lambda k: str(k).zfill(prefix_width))
    )

    epoch_labels = epoch_prefixes + "_" + epoch_date_strings
    epoch_label_map = epoch_labels.to_dict()
    df["epoch_id"] = df["epoch_id_int"].map(epoch_label_map)

    # df["epoch_id"] = pd.Categorical(
    #     df["epoch_id"], categories=epoch_labels.values, ordered=True
    # )

    df = df.drop(["mid_time_temp"], axis=1)

    return df


# TODO: can we do better than this? It's ugly
# TODO: the matplot ui uses this to recalculate on the fly for display - ditch this
def epochs_from_time_delta(
    obs_log: SwiftUvotObservationLogDataframe, max_time_between_obs: u.Quantity
) -> list[SwiftUvotObservationLogDataframe]:
    """
    Takes an observation log and slices it into a list of observation logs, each representing a time slice
    where observations are no more than max_time_between_obs apart.
    """
    # sort observations by time, oldest first
    obs_log = obs_log.sort_values(by="MID_TIME", ascending=True).reset_index(drop=True)

    epoch_list = []
    epoch_count = 0

    while True:
        max_index = len(obs_log) - 1

        t_start = Time(obs_log.iloc[0].MID_TIME) - 1 * u.s  # type: ignore

        # keep checking if next observation is within max_time_delta
        prev_index = 0
        while True:
            prev_time = Time(obs_log.iloc[prev_index].MID_TIME)
            if max_index == 0:
                # this is the only row left, so set t_end and break
                t_end = prev_time + 1 * u.s  # type: ignore
                break

            cur_index = prev_index + 1

            cur_time = Time(obs_log.iloc[cur_index].MID_TIME)
            delta_t = cur_time - prev_time

            # is the time delta to the next observation too large?  Use the previous as the stopping point
            if delta_t > max_time_between_obs:
                t_end = prev_time + 1 * u.s  # type: ignore
                break

            # is the current index the last row?  Use this last row as the stopping point
            if cur_index == max_index:
                t_end = cur_time + 1 * u.s  # type: ignore
                break

            prev_index = cur_index

        t_start, t_end = t_start.to_datetime(), t_end.to_datetime()
        time_filter = (obs_log.MID_TIME > t_start) & (obs_log.MID_TIME < t_end)

        # TODO: instead of slicing, just assign obs_log[time_filter].epoch_id = epoch_count

        # slice a copy of the observation log and convert into an epoch
        epoch = obs_log[time_filter].copy()
        epoch_list.append(epoch.reset_index(drop=True))
        epoch_count += 1

        cutoff_mask = obs_log.MID_TIME > t_end
        obs_log = obs_log[cutoff_mask]  # type: ignore

        # check to see if there is any data left
        if obs_log.empty:
            break

    return epoch_list
