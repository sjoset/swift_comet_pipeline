import copy
import numpy as np

from enum import Enum, auto
from dataclasses import dataclass

from astropy.visualization import ZScaleInterval

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from photutils.aperture import (
    CircularAperture,
    ApertureStats,
)

from swift_types import SwiftUVOTImage


__all__ = ["BackgroundDeterminationMethod", "BackgroundResult", "determine_background"]


class BackgroundDeterminationMethod(str, Enum):
    swift_constant = auto()
    manual_aperture = auto()
    gui_manual_aperture = auto()
    walking_aperture_ensemble = auto()


@dataclass
class BackgroundResult:
    count_rate_per_pixel: float
    sigma: float


def determine_background(
    img: SwiftUVOTImage,
    background_method: BackgroundDeterminationMethod,
    **kwargs,
) -> BackgroundResult:
    if background_method == BackgroundDeterminationMethod.swift_constant:
        return bg_swift_constant(img=img)
    elif background_method == BackgroundDeterminationMethod.manual_aperture:
        return bg_manual_aperture(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
        return bg_walking_aperture_ensemble(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        return bg_gui_manual_aperture(img=img, **kwargs)


# TODO: come up with some decent values here
def bg_swift_constant(img: SwiftUVOTImage) -> BackgroundResult:  # pyright: ignore
    return BackgroundResult(count_rate_per_pixel=1.0, sigma=1.0)


# TODO: sigma clipping?
def bg_manual_aperture(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> BackgroundResult:
    # make the aperture as a one-element list to placate the type checker
    background_aperture = CircularAperture(
        [(aperture_x, aperture_y)], r=aperture_radius
    )

    aperture_stats = ApertureStats(img, background_aperture)
    count_rate_per_pixel = aperture_stats.median[0]

    return BackgroundResult(
        count_rate_per_pixel=count_rate_per_pixel, sigma=aperture_stats.std[0]
    )


# TODO: add dataclasses for walking ensemble input parameters
def bg_walking_aperture_ensemble(
    img: SwiftUVOTImage,  # pyright: ignore
) -> BackgroundResult:
    return BackgroundResult(count_rate_per_pixel=2.0, sigma=2.0)


class BackgroundAperturePlacementPlot(object):
    def __init__(self, img, title="aoeu"):
        self.original_img = copy.deepcopy(img)
        self.bg_count_rate = 0

        self.img = img
        self.title = title

        self.image_center_x = int(np.floor(img.shape[1] / 2))
        self.image_center_y = int(np.floor(img.shape[0] / 2))

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        # self.ax.axis([0, 100, 0, 100])
        self.ax.set_aspect("equal")  # type: ignore

        self.radius_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="orange")
        self.radius_slider = Slider(
            self.radius_slider_ax, "r", valmin=1, valmax=100, valstep=1, valinit=50
        )
        self.radius_slider.on_changed(self.onslider)

        self.aperture = plt.Circle(
            (self.image_center_x, self.image_center_y),
            edgecolor="white",
            fill=False,
            alpha=0.3,
        )
        self.ax.add_patch(self.aperture)  # type: ignore

        self.aperture.set_radius(self.radius_slider.val)

        self.colormap = "magma"
        self.zscale = ZScaleInterval()
        self.vmin, self.vmax = self.zscale.get_limits(self.img)
        self.img_plot = self.ax.imshow(  # type: ignore
            self.img, vmin=self.vmin, vmax=self.vmax, origin="lower", cmap=self.colormap
        )

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        plt.draw()

    def onclick(self, event):
        # only move the aperture if they click on the image
        if event.inaxes != self.ax:
            return
        self.aperture.center = event.xdata, event.ydata
        self.recalc_background()
        self.fig.canvas.draw_idle()  # type: ignore

    def recalc_background(self):
        self.bg_count_rate = bg_manual_aperture(
            img=self.original_img,
            aperture_x=self.aperture.get_center()[0],
            aperture_y=self.aperture.get_center()[1],
            aperture_radius=self.aperture.radius,
        ).count_rate_per_pixel
        self.img_plot.set_data(self.original_img - self.bg_count_rate)

    def onslider(self, new_value):
        self.aperture.set_radius(new_value)
        self.recalc_background()
        self.fig.canvas.draw_idle()  # type: ignore

    def show(self):
        plt.show()


def bg_gui_manual_aperture(img: SwiftUVOTImage):
    bg = BackgroundAperturePlacementPlot(img)
    bg.show()

    return bg_manual_aperture(
        img=img,
        aperture_x=bg.aperture.get_center()[0],
        aperture_y=bg.aperture.get_center()[1],
        aperture_radius=bg.aperture.radius,
    )
