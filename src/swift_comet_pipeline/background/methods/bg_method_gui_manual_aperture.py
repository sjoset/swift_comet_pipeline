import copy

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from matplotlib.widgets import Slider

from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.background.methods.bg_method_aperture import (
    bg_manual_aperture_mean,
    bg_manual_aperture_median,
)


def bg_gui_manual_aperture(img: SwiftUVOTImage, filter_type: SwiftFilter):
    bg = BackgroundAperturePlacementPlot(img, filter_type=filter_type)
    bg.show()

    # return bg_manual_aperture_mean(
    #     img=img,
    #     aperture_x=bg.aperture.get_center()[0],  # type: ignore
    #     aperture_y=bg.aperture.get_center()[1],  # type: ignore
    #     aperture_radius=bg.aperture.radius,
    # )

    ap_x = float(bg.aperture.get_center()[0])  # type: ignore
    ap_y = float(bg.aperture.get_center()[1])  # type: ignore
    ap_radius = float(bg.aperture.radius)

    params = {
        "aperture_x": ap_x,
        "aperture_y": ap_y,
        "aperture_radius": ap_radius,
    }

    return BackgroundResult(
        count_rate_per_pixel=bg_manual_aperture_median(
            img=img, aperture_x=ap_x, aperture_y=ap_y, aperture_radius=ap_radius
        ),
        params=params,
        method=BackgroundDeterminationMethod.gui_manual_aperture,
    )


class BackgroundAperturePlacementPlot:
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
        bgresult = bg_manual_aperture_mean(
            img=self.original_img,
            aperture_x=self.aperture.get_center()[0],  # type: ignore
            aperture_y=self.aperture.get_center()[1],  # type: ignore
            aperture_radius=self.aperture.radius,
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
