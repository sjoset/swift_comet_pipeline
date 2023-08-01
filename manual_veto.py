#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from epochs import Epoch
from swift_types import SwiftData

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from mpl_toolkits import axes_grid1


__all__ = ["manual_veto"]


class EpochImageSlider(Slider):
    def __init__(self, ax, num_images, valfmt="%1d", **kwargs):
        self.facecolor = kwargs.get("facecolor", "w")
        self.activecolor = kwargs.pop("activecolor", "b")
        self.fontsize = kwargs.pop("fontsize", 10)
        self.num_images = num_images
        self.label = "Image"
        initial_image_index = 0

        super(EpochImageSlider, self).__init__(
            ax=ax,
            label=self.label,
            valmin=0,
            valmax=num_images,
            valinit=initial_image_index,
            valfmt=valfmt,
            **kwargs,
        )

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []

        for i in range(num_images):
            facecolor = self.activecolor if i == initial_image_index else self.facecolor
            r = Rectangle(
                xy=(float(i) / num_images, 0),
                width=1.0 / num_images,
                height=1.0,
                transform=ax.transAxes,
                facecolor=facecolor,
            )
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(
                x=float(i) / num_images + 0.5 / num_images,
                y=0.5,
                s=str(i),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.fontsize,
            )
        self.valtext.set_visible(False)

        divider = axes_grid1.make_axes_locatable(ax)
        prev_button_axis = divider.append_axes("right", size="5%", pad=0.05)
        forward_button_axis = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = Button(
            ax=prev_button_axis,
            label="prev",
            color=self.facecolor,
            hovercolor=self.activecolor,
        )
        self.button_forward = Button(
            ax=forward_button_axis,
            label="next",
            color=self.facecolor,
            hovercolor=self.activecolor,
        )
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.previous_image)
        self.button_forward.on_clicked(self.next_image)

    def _update(self, event):
        super(EpochImageSlider, self)._update(event)
        i = int(self.val)
        if i >= self.num_images:
            return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.num_images):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def next_image(self, event):
        image_index = int(self.val) + 1
        if (image_index < self.valmin) or (image_index >= self.num_images):
            return
        self.set_val(image_index)
        self._colorize(image_index)

    def previous_image(self, event):
        image_index = int(self.val) - 1
        if (image_index < self.valmin) or (image_index >= self.num_images):
            return
        self.set_val(image_index)
        self._colorize(image_index)


class EpochImagePlot(object):
    def __init__(self, swift_data: SwiftData, epoch: Epoch, epoch_title: str):
        self.swift_data = swift_data
        self.epoch = epoch
        self.num_images = len(epoch)
        self.epoch_title = epoch_title

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.18)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)  # type: ignore

        self.zscale = ZScaleInterval()

        self.slider_ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.04])
        self.slider = EpochImageSlider(
            self.slider_ax, self.num_images, activecolor="orange"
        )
        self.slider.on_changed(self.slider_update)

        self.veto_ax = plt.axes([0.7, 0.025, 0.1, 0.04])
        self.approve_ax = plt.axes([0.85, 0.025, 0.1, 0.04])

        self.veto_button = Button(self.veto_ax, "Veto", color="red", hovercolor="0.975")
        self.veto_button.on_clicked(self.veto_current_image)
        self.approve_button = Button(
            self.approve_ax, "Approve", color="green", hovercolor="0.975"
        )
        self.approve_button.on_clicked(self.approve_current_image)

        self.colorbar_axis = axes_grid1.make_axes_locatable(self.ax).append_axes(
            "right", size="5%", pad="2%"
        )

        self.img_plot = None
        self.current_image_index = -1
        self.current_image = None
        self.do_plot(0)

    def get_image_by_index(self, index):
        if index < 0 or index >= self.num_images:
            return
        epoch_row = self.epoch.iloc[self.current_image_index]

        image_path = (
            self.swift_data.get_uvot_image_directory(obsid=epoch_row.OBS_ID)
            / epoch_row.FITS_FILENAME
        )
        return fits.getdata(image_path, ext=epoch_row.EXTENSION)  # type: ignore

    def do_plot(self, index):
        if index < 0 or index >= self.num_images:
            return

        self.current_image_index = index
        epoch_row = self.epoch.iloc[self.current_image_index]

        self.image_path = (
            self.swift_data.get_uvot_image_directory(obsid=epoch_row.OBS_ID)
            / epoch_row.FITS_FILENAME
        )
        self.current_image = fits.getdata(self.image_path, ext=epoch_row.EXTENSION)  # type: ignore
        self.vmin, self.vmax = self.zscale.get_limits(self.current_image)

        if self.img_plot is None:
            self.img_plot = self.ax.imshow(self.current_image, vmin=self.vmin, vmax=self.vmax, origin="lower", cmap=self.colormap_by_veto())  # type: ignore
            self.colorbar = self.fig.colorbar(self.img_plot, cax=self.colorbar_axis)
        else:
            self.img_plot.set_data(self.current_image)
            self.img_plot.set_cmap(self.colormap_by_veto())
            self.img_plot.set_clim(vmin=self.vmin, vmax=self.vmax)

        self.ax.set_title(  # type: ignore
            self.epoch_title
            + "  ("
            + self.image_path.name
            + "  extension "
            + str(epoch_row.EXTENSION)
            + ")  "
            + f"{epoch_row.EXPOSURE:4.1f}"
            + " s exposure"
        )

    def redraw_image(self):
        self.do_plot(self.current_image_index)
        plt.draw()

    def slider_update(self, new_index):
        if int(new_index) == self.current_image_index:
            return
        self.do_plot(int(new_index))
        return

    def veto_current_image(self, event):
        self.epoch.loc[self.current_image_index, "manual_veto"] = np.True_
        self.redraw_image()

    def approve_current_image(self, event):
        self.epoch.loc[self.current_image_index, "manual_veto"] = np.False_
        self.redraw_image()

    def colormap_by_veto(self):
        if self.epoch.loc[self.current_image_index, "manual_veto"] is np.True_:
            cmap = "binary"
        else:
            cmap = "magma"
        return cmap

    def on_key_press(self, event):
        if event.key == "l":
            if self.current_image_index < (self.num_images - 1):
                self.slider.next_image(event)
        elif event.key == "h":
            if self.current_image_index > 0:
                self.slider.previous_image(event)
        elif event.key == "v":
            self.veto_current_image(event)
        elif event.key == "a":
            self.approve_current_image(event)

    def show(self):
        plt.show()


def manual_veto(swift_data: SwiftData, epoch: Epoch, epoch_title: str) -> Epoch:
    # group the uvv and uw1 filters together for image viewing
    epoch = epoch.sort_values("FILTER").reset_index(drop=True)
    EpochImagePlot(swift_data=swift_data, epoch=epoch, epoch_title=epoch_title).show()
    epoch = epoch.sort_values("DATE_OBS").reset_index(drop=True)
    return epoch
