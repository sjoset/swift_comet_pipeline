import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.pyplot import cm

from swift_comet_pipeline.observationlog.observation_log import SwiftObservationLog
from swift_comet_pipeline.observationlog.slice_observation_log_into_epochs import (
    epochs_from_time_delta,
)


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


def gui_select_epoch_time_window(obs_log: SwiftObservationLog) -> u.Quantity:
    quantity_support()
    etws = EpochTimeWindowSelect(obs_log)
    etws.show()
    return etws.dt
