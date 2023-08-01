#!/usr/bin/env python3

import sys
import numpy as np
import pathlib
from astropy.visualization import ZScaleInterval
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


# def demo():
#     fig, ax = plt.subplots()
#     plt.subplots_adjust(left=0.25, bottom=0.25)
#     ax.axis([0, 100, 0, 100])
#     ax.set_aspect("equal")
#
#     axcolor = "skyblue"
#     sl1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
#     sl2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#     # sl3 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
#
#     slider_r1 = Slider(sl1, "r1", 0.0, 50.0, 25)
#     slider_r2 = Slider(sl2, "r2", 0.0, 50.0, 25)
#     # slider_d = Slider(sl3, "dist", 0.0, 100.0, 50)
#
#     circ1 = plt.Circle((25, 50), 25, ec="k")
#     circ2 = plt.Circle((75, 50), 25, ec="k")
#     ax.add_patch(circ1)
#     ax.add_patch(circ2)
#
#     def update(val):
#         r1 = slider_r1.val
#         r2 = slider_r2.val
#         # d = slider_d.val
#         # circ1.center = 50 - d / 2.0, 50
#         # circ2.center = 50 + d / 2.0, 50
#         circ1.set_radius(r1)
#         circ2.set_radius(r2)
#         fig.canvas.draw_idle()
#
#     slider_r1.on_changed(update)
#     slider_r2.on_changed(update)
#     # slider_d.on_changed(update)
#
#     def onclick(event):
#         print(f"{circ1.center=}")
#         circ1.center = event.xdata, event.ydata
#         fig.canvas.draw_idle()
#         print(f"x: {event.xdata}, y: {event.ydata}")
#
#     fig.canvas.mpl_connect("button_press_event", onclick)
#
#     plt.show()


class AperturePlacementPlot(object):
    def __init__(self, img, title="aoeu"):
        self.img = img
        self.title = title

        self.image_center_x = int(np.floor(img.shape[1] / 2))
        self.image_center_y = int(np.floor(img.shape[0] / 2))

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        # self.ax.axis([0, 100, 0, 100])
        self.ax.set_aspect("equal")

        self.radius_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="orange")
        self.radius_slider = Slider(
            self.radius_slider_ax, "r", valmin=1, valmax=100, valstep=1, valinit=50
        )
        self.radius_slider.on_changed(self.onslider)

        self.aperture = plt.Circle(
            (self.image_center_x, self.image_center_y), edgecolor="black", fill=False
        )
        self.ax.add_patch(self.aperture)

        self.aperture.set_radius(self.radius_slider.val)

        self.colormap = "magma"
        self.zscale = ZScaleInterval()
        self.vmin, self.vmax = self.zscale.get_limits(self.img)
        self.img_plot = self.ax.imshow(
            self.img, vmin=self.vmin, vmax=self.vmax, origin="lower", cmap=self.colormap
        )

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.draw()

    def onclick(self, event):
        print(f"{event.inaxes=}")
        # only move the aperture if they click on the image
        if event.inaxes != self.ax:
            return
        self.aperture.center = event.xdata, event.ydata
        print(self.aperture.center)
        self.fig.canvas.draw_idle()

    def onslider(self, new_value):
        self.aperture.set_radius(new_value)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def get_background_aperture


def main():
    fits_path = pathlib.Path("stacked/000_2014_14_Aug_uvv_sum.fits")
    image_data = fits.getdata(fits_path)
    bg = AperturePlacementPlot(image_data)
    bg.show()
    print(bg.aperture.get_center(), bg.aperture.radius)


if __name__ == "__main__":
    sys.exit(main())
