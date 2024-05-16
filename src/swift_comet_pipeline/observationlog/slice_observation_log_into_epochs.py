from typing import List

import astropy.units as u
from astropy.time import Time

from swift_comet_pipeline.observationlog.observation_log import SwiftObservationLog
from swift_comet_pipeline.observationlog.epoch import Epoch


def epochs_from_time_delta(
    obs_log: SwiftObservationLog, max_time_between_obs: u.Quantity
) -> List[Epoch]:
    # sort observations by time, oldest first
    obs_log = obs_log.sort_values(by="MID_TIME", ascending=True).reset_index(drop=True)

    # num_observations = len(obs_log)
    epoch_list = []
    epoch_count = 0

    while True:
        max_index = len(obs_log) - 1

        t_start = Time(obs_log.iloc[0].MID_TIME) - 1 * u.s

        # keep checking if next observation is within max_time_delta
        prev_index = 0
        while True:
            prev_time = Time(obs_log.iloc[prev_index].MID_TIME)
            if max_index == 0:
                # this is the only row left, so set t_end and break
                t_end = prev_time + 1 * u.s
                break

            cur_index = prev_index + 1

            cur_time = Time(obs_log.iloc[cur_index].MID_TIME)
            delta_t = cur_time - prev_time

            # is the time delta to the next observation too large?  Use the previous as the stopping point
            if delta_t > max_time_between_obs:
                t_end = prev_time + 1 * u.s
                break

            # is the current index the last row?  Use this last row as the stopping point
            if cur_index == max_index:
                t_end = cur_time + 1 * u.s
                break

            prev_index = cur_index

        t_start, t_end = t_start.to_datetime(), t_end.to_datetime()
        time_filter = (obs_log.MID_TIME > t_start) & (obs_log.MID_TIME < t_end)

        # slice a copy of the observation log and convert into an epoch
        epoch = obs_log[time_filter].copy()
        epoch_list.append(epoch.reset_index(drop=True))
        epoch_count += 1
        # print(f"Epoch {epoch_count} --> {len(epoch)} observations")

        cutoff_mask = obs_log.MID_TIME > t_end
        obs_log = obs_log[cutoff_mask]  # type: ignore

        # check to see if there is any data left
        if obs_log.empty:
            break

    # print(f"Total observations in observation log before slicing: {num_observations}")
    # obs_sum = sum([len(x) for x in epoch_list])
    # print(f"Total observations in all epochs after slicing: {obs_sum}")
    return epoch_list
