import copy
import numpy as np

from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from types import SimpleNamespace

from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from photutils.aperture import (
    CircularAperture,
    ApertureStats,
)

from swift_comet_pipeline.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.count_rate import CountRatePerPixel


__all__ = [
    "BackgroundDeterminationMethod",
    "BackgroundResult",
    "determine_background",
    "background_result_to_dict",
    "dict_to_background_result",
]


class BackgroundDeterminationMethod(StrEnum):
    swift_constant = auto()
    manual_aperture_mean = auto()
    manual_aperture_median = auto()
    gui_manual_aperture = auto()
    walking_aperture_ensemble = auto()


@dataclass
class BackgroundResult:
    count_rate_per_pixel: CountRatePerPixel
    params: dict
    method: BackgroundDeterminationMethod


def determine_background(
    img: SwiftUVOTImage,
    background_method: BackgroundDeterminationMethod,
    **kwargs,
) -> BackgroundResult:
    """The optional dict is to return any details about the method that was used"""
    if background_method == BackgroundDeterminationMethod.swift_constant:
        return bg_swift_constant(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.manual_aperture_mean:
        return bg_manual_aperture_mean(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.manual_aperture_median:
        return bg_manual_aperture_median(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.walking_aperture_ensemble:
        return bg_walking_aperture_ensemble(img=img, **kwargs)
    elif background_method == BackgroundDeterminationMethod.gui_manual_aperture:
        return bg_gui_manual_aperture(img=img, **kwargs)


# TODO: come up with some decent values here
def bg_swift_constant(
    img: SwiftUVOTImage, filter_type: SwiftFilter
) -> BackgroundResult:
    """Return what the background is believed to be based on the published information about the SWIFT instrumentation"""
    if filter_type == SwiftFilter.uvv:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)
    elif filter_type == SwiftFilter.uvv:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)
    else:
        count_rate_per_pixel = CountRatePerPixel(value=1.0, sigma=1.0)

    params = {}
    return BackgroundResult(
        count_rate_per_pixel=count_rate_per_pixel,
        params=params,
        method=BackgroundDeterminationMethod.swift_constant,
    )


# TODO: sigma clipping?
def bg_manual_aperture_stats(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> ApertureStats:
    # make the aperture as a one-element list to placate the type checker
    background_aperture = CircularAperture(
        [(aperture_x, aperture_y)], r=aperture_radius
    )

    aperture_stats = ApertureStats(img, background_aperture)

    return aperture_stats


def bg_manual_aperture_median(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> BackgroundResult:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_x=aperture_x,
        aperture_y=aperture_y,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.median[0]
    # error of median is a factor larger than sigma
    error_abs = 1.2533 * aperture_stats.std[0]

    params = {
        "aperture_x": float(aperture_x),
        "aperture_y": float(aperture_y),
        "aperture_radius": float(aperture_radius),
    }

    return BackgroundResult(
        count_rate_per_pixel=CountRatePerPixel(
            value=count_rate_per_pixel, sigma=error_abs
        ),
        params=params,
        method=BackgroundDeterminationMethod.manual_aperture_median,
    )


def bg_manual_aperture_mean(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> BackgroundResult:
    aperture_stats = bg_manual_aperture_stats(
        img=img,
        aperture_x=aperture_x,
        aperture_y=aperture_y,
        aperture_radius=aperture_radius,
    )

    count_rate_per_pixel = aperture_stats.mean[0]
    error_abs = aperture_stats.std[0]

    params = {
        "aperture_x": float(aperture_x),
        "aperture_y": float(aperture_y),
        "aperture_radius": float(aperture_radius),
    }

    return BackgroundResult(
        count_rate_per_pixel=CountRatePerPixel(
            value=count_rate_per_pixel, sigma=error_abs
        ),
        params=params,
        method=BackgroundDeterminationMethod.manual_aperture_mean,
    )


# TODO: add dataclass for walking ensemble input parameters?
def bg_walking_aperture_ensemble(
    img: SwiftUVOTImage,  # pyright: ignore
) -> BackgroundResult:
    return BackgroundResult(
        count_rate_per_pixel=CountRatePerPixel(value=2.0, sigma=2.0),
        params={},
        method=BackgroundDeterminationMethod.walking_aperture_ensemble,
    )


class BackgroundAperturePlacementPlot(object):
    def __init__(self, img, filter_type: SwiftFilter):
        self.original_img = copy.deepcopy(img)
        self.bg_count_rate = 0

        self.img = img
        self.filter_type = filter_type
        self.title = (
            f"Determine background for filter {filter_to_file_string(filter_type)}"
        )

        self.image_center_x = int(np.floor(img.shape[1] / 2))
        self.image_center_y = int(np.floor(img.shape[0] / 2))

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)
        # self.ax.axis([0, 100, 0, 100])
        self.ax.set_aspect("equal")  # type: ignore
        self.ax.set_title(self.title)  # type: ignore

        self.count_rate_annotation = self.ax.annotate(  # type: ignore
            self.count_rate_string(),
            xy=(0.03, 0.95),
            xytext=(0.03, 0.95),
            textcoords="axes fraction",
            size=10,
            color="orange",
        )

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

    def count_rate_string(self):
        return f"Background: {self.bg_count_rate:07.6f} counts per second per pixel"

    def recalc_background(self):
        bgresult = bg_manual_aperture_mean(
            img=self.original_img,
            aperture_x=self.aperture.get_center()[0],
            aperture_y=self.aperture.get_center()[1],
            aperture_radius=self.aperture.radius,
        )
        self.bg_count_rate = bgresult.count_rate_per_pixel.value
        self.count_rate_annotation.set_text(self.count_rate_string())
        self.img_plot.set_data(self.original_img - self.bg_count_rate)

    def onslider(self, new_value):
        self.aperture.set_radius(new_value)
        self.recalc_background()
        self.fig.canvas.draw_idle()  # type: ignore

    def show(self):
        plt.show()


def bg_gui_manual_aperture(img: SwiftUVOTImage, filter_type: SwiftFilter):
    bg = BackgroundAperturePlacementPlot(img, filter_type=filter_type)
    bg.show()

    return bg_manual_aperture_mean(
        img=img,
        aperture_x=bg.aperture.get_center()[0],
        aperture_y=bg.aperture.get_center()[1],
        aperture_radius=bg.aperture.radius,
    )


def background_result_to_dict(
    bg_result: BackgroundResult,
) -> dict:
    # yaml serializer doesn't support numpy floats for some reason
    bg_result.count_rate_per_pixel.value = float(bg_result.count_rate_per_pixel.value)
    bg_result.count_rate_per_pixel.sigma = float(bg_result.count_rate_per_pixel.sigma)

    bg_dict = {
        "params": bg_result.params,
        "count_rate_per_pixel": asdict(bg_result.count_rate_per_pixel),
        "method": str(bg_result.method),
    }

    return bg_dict


def dict_to_background_result(raw_yaml: dict) -> BackgroundResult:
    bg = SimpleNamespace(**raw_yaml)
    return BackgroundResult(
        CountRatePerPixel(**bg.count_rate_per_pixel),
        params=bg.params,
        method=BackgroundDeterminationMethod(bg.method),
    )
