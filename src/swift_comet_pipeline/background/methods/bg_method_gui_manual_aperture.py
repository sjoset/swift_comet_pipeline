import copy

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.widgets import Slider

from swift_comet_pipeline.background.methods.bg_method_aperture import bg_in_aperture
from swift_comet_pipeline.swift.swift_filter_to_string import filter_to_file_string
from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    BackgroundValueEstimator,
)
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def bg_gui_manual_aperture(
    img: SwiftUVOTImage, filter_type: SwiftFilter, exposure_time_s: float
):
    bg = BackgroundAperturePlacementPlot(
        img=img, filter_type=filter_type, exposure_time_s=exposure_time_s
    )
    bg.show()

    ap_x = float(bg.aperture.get_center()[0])  # type: ignore
    ap_y = float(bg.aperture.get_center()[1])  # type: ignore
    ap_radius = float(bg.aperture.radius)
    ap_area = int(np.round(np.pi * ap_radius**2))

    params = {
        "aperture_x": ap_x,
        "aperture_y": ap_y,
        "aperture_radius": ap_radius,
    }

    return BackgroundResult(
        count_rate_per_pixel=bg_in_aperture(
            img=img,
            aperture_center=PixelCoord(x=ap_x, y=ap_y),
            aperture_radius=ap_radius,
            bg_estimator=BackgroundValueEstimator.median,
            exposure_time_s=exposure_time_s,
        ),
        bg_estimator=BackgroundValueEstimator.median,
        bg_aperture_area=ap_area,
        params=params,
        method=BackgroundDeterminationMethod.gui_manual_aperture,
    )


class BackgroundAperturePlacementPlot:
    def __init__(
        self, img: SwiftUVOTImage, filter_type: SwiftFilter, exposure_time_s: float
    ):
        self.original_img = copy.deepcopy(img)
        self.bg_count_rate = 0

        self.img = img
        self.filter_type = filter_type
        self.exposure_time_s = exposure_time_s
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

        self.radius_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="orange")  # type: ignore
        self.radius_slider = Slider(
            self.radius_slider_ax, "r", valmin=1, valmax=100, valstep=1, valinit=50
        )
        self.radius_slider.on_changed(self.onslider)

        self.aperture = plt.Circle(  # type: ignore
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
        aperture_center = PixelCoord(
            x=self.aperture.get_center()[0], y=self.aperture.get_center()[1]  # type: ignore
        )
        bgresult = bg_in_aperture(
            img=self.original_img,
            aperture_center=aperture_center,
            aperture_radius=self.aperture.radius,
            bg_estimator=BackgroundValueEstimator.median,
            exposure_time_s=self.exposure_time_s,
        )
        self.bg_count_rate = bgresult.value
        self.count_rate_annotation.set_text(self.count_rate_string())
        self.img_plot.set_data(self.original_img - self.bg_count_rate)

    def onslider(self, new_value):
        self.aperture.set_radius(new_value)
        self.recalc_background()
        self.fig.canvas.draw_idle()  # type: ignore

    def show(self):
        plt.show()
