# import calendar
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.visualization import quantity_support

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.pyplot import cm

from typing import List

# from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.observationlog.observation_log import SwiftObservationLog
from swift_comet_pipeline.observationlog.epochs import epoch_from_obs_log


__all__ = [
    "select_epoch_time_window",
    "epochs_from_time_delta",
    # "time_delta_loop",
]


def epochs_from_time_delta(
    obs_log: SwiftObservationLog, max_time_between_obs: u.Quantity
) -> List[SwiftObservationLog]:
    # sort by observations by time, oldest first
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
        epoch = epoch_from_obs_log(epoch)  # type: ignore
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


# def time_delta_loop(obs_log: SwiftObservationLog) -> List[SwiftObservationLog]:
#     epoch_list = []
#
#     finished = False
#     time_units = [u.hour, u.day, u.s]
#
#     num_observations = len(obs_log)
#     print(f"Total number of uw1 or uvv observations in dataset: {num_observations}")
#
#     while finished is False:
#         time_delta_raw = get_float("Enter max time delta: ")
#         time_unit = time_units[get_selection(time_units)]
#         time_delta = time_delta_raw * time_unit
#
#         epoch_list = epochs_from_time_delta(
#             obs_log=obs_log, max_time_between_obs=time_delta
#         )
#         print(f"{len(epoch_list)} epochs identified with time delta of {time_delta}:")
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


class EpochTimeWindowSelect(object):
    def __init__(self, obs_log: SwiftObservationLog):
        self.obs_log = obs_log.copy()

        # self.min_time = (Time(np.min(self.obs_log.MID_TIME)) - 1 * u.day).to_datetime()  # type: ignore
        # self.max_time = (Time(np.max(self.obs_log.MID_TIME)) + 1 * u.day).to_datetime()  # type: ignore

        initial_dt = 12 * u.hour  # type: ignore
        self.fig, self.ax = plt.subplots(1, 1)

        # self.ax.set_xlim(self.min_time, self.max_time)  # type: ignore
        self.ax.set_ylim(-0.05, 0.05)  # type: ignore
        self.ax.get_yaxis().set_visible(False)  # type: ignore
        self.fig.subplots_adjust(bottom=0.2)

        self.slider_ax = self.fig.add_axes([0.3, 0.1, 0.6, 0.04])  # type: ignore
        self.slider = Slider(
            ax=self.slider_ax,
            label="Max time between observations (hours)",
            valmin=1.0,
            valmax=96.0,
            valstep=1.0,
            valinit=initial_dt.to_value(u.hour),
        )
        self.slider.on_changed(self.slider_update)

        self.point_colormap = cm.twilight  # type: ignore
        self.bar_colormap = cm.twilight_shifted  # type: ignore

        self.plot_active = False
        self.do_plot(initial_dt)

    def do_plot(self, dt):
        self.dt = dt
        epoch_list = epochs_from_time_delta(self.obs_log.copy(), self.dt)

        point_colors = iter(self.point_colormap(np.linspace(0, 1, len(epoch_list))))
        bar_colors = iter(self.bar_colormap(np.linspace(0, 1, len(epoch_list))))

        if self.plot_active is True:
            self.ax.clear()  # type: ignore
        else:
            self.plot_active = True

        for i, (epoch, point_color, bar_color) in enumerate(
            zip(epoch_list, point_colors, bar_colors)
        ):
            ts = epoch.MID_TIME
            min_t, max_t = np.min(ts), np.max(ts)
            ys = np.zeros(len(ts))
            self.ax.scatter(ts, ys, color=point_color)  # type: ignore
            self.ax.axvspan(min_t, max_t, color=bar_color, alpha=0.2)  # type: ignore
            self.ax.text(
                x=min_t,
                y=(0.02 + float(i) * 0.03 / len(epoch_list)),
                s=f"{i+1}",
                color="black",
            )
            self.ax.text(
                x=min_t,
                y=(0.02 + float(i) * 0.03 / len(epoch_list)) * -1,
                s=f"{len(epoch)} obs",
                color="black",
            )

        self.ax.set_title(  # type: ignore
            f"Time difference of {self.dt} results in {len(epoch_list)} epochs"
        )

    def slider_update(self, new_dt_hours):
        if int(new_dt_hours) == self.dt.to_value(u.hour):
            return
        self.do_plot(int(new_dt_hours) * u.hour)  # type: ignore
        return

    def show(self):
        plt.show()


def select_epoch_time_window(obs_log: SwiftObservationLog) -> u.Quantity:
    quantity_support()
    etws = EpochTimeWindowSelect(obs_log)
    etws.show()
    return etws.dt
