import itertools
from typing import List

import astropy.units as u
from astropy.time import Time

from swift_comet_pipeline.observationlog.observation_log import SwiftObservationLog
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.swift.uvot_image import SwiftImageDataMode


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

    return epoch_list


def split_epoch_list_into_data_and_event_epochs(epoch_list: list[Epoch]) -> list[Epoch]:
    """
    Takes a list of epochs, and for each epoch, replaces it with a pair of epochs that include only
    data mode or event mode images.
    If the epoch does not contain either mode of data, it will disappear from the list.

    """

    # TODO: If the entire dataset has no data mode or event mode images, we should check for this and return None

    split_epoch_list = []
    for epoch in epoch_list:
        split_epoch_list.append(split_epoch_into_data_and_event_epochs(epoch=epoch))

    return list(itertools.chain.from_iterable(split_epoch_list))


def split_epoch_into_data_and_event_epochs(epoch: Epoch) -> list[Epoch]:
    """
    Returns a list of two epochs: one epoch of only data mode images, the other of only event mode images.
    List may be empty if epoch contains neither.
    """

    epoch_list = []
    for datamode in [SwiftImageDataMode.data_mode, SwiftImageDataMode.event_mode]:
        num_imgs = epoch.DATAMODE.value_counts().get(datamode, 0)
        if num_imgs == 0:
            continue
        split_epoch = epoch[epoch.DATAMODE == datamode].copy()
        epoch_list.append(split_epoch)

    return epoch_list
