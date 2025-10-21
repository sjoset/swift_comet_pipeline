import copy

from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.photometry.comet.extract_comet_radial_profile import (
    calculate_distance_from_center_mesh,
    extract_comet_radial_median_profile_from_cone,
    radial_profile_to_image,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.scp_types.compound.background_result import BackgroundResult
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive import *


class RadialProfileSelectionPlot(object):
    # TODO: make this take img_axis and profile_axis as arguments to embed in other figures
    # TODO: draw reference profiles after showing image
    # TODO: add buttons/key input to snap to reference profiles
    # TODO: add buttons/key to increase radius without changing angle
    # TODO: add position angles for sun/velocity
    # TODO: document
    def __init__(
        self,
        eid: EpochIndexEntry,
        img: SwiftUvotImage,
        bgr: BackgroundResult,
        reference_profiles: (
            dict[UvotFilter, CometRadialProfileFromConicalRegion] | None
        ),
        filter_type: UvotFilter,
    ):
        self.epoch_summary = eid
        self.img = img
        self.filter_type = filter_type
        self.bgr = bgr
        self.bg_shot_noise_error = np.sqrt(self.bgr.bg_shot_noise_variance)

        self.reference_profiles = reference_profiles or {}

        self.image_center = get_uvot_image_center(self.img)
        self.time_from_perihelion = eid.time_from_perihelion

        # figure, axes for each graph
        self.create_basic_mpl_elements()
        # for displaying the cone over the image
        self.create_cone_slider_mpl_elements()
        # for displaying the stacked image
        self.create_image_mpl_elements()

        # TODO: move this to its own create_() function, along with the profile_plot
        # holds the plots for 2d extracted profiles
        self.profile_plot = None
        # Image coordinates for extracting the profile: start at comet center, and stop at arbitrary point away from center for initialization
        self.profile_begin: PixelCoord = self.image_center
        self.profile_end: PixelCoord = PixelCoord(
            x=self.image_center.x + 50, y=self.image_center.y + 50
        )

        # for the lines showing the extraction cone
        self.create_profile_extraction_mpl_elements()

        # for quickly extracting radial profiles
        self.setup_mesh()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        self.update_plots()

    def create_basic_mpl_elements(self):
        # self.fig, self.axes = plt.subplots(1, 4)
        self.fig, self.axes = plt.subplot_mosaic(
            # [["profile", "profile", "profile"], ["cone", "subtracted", "profile_img"]]
            [["cone", "cone", "cone"], ["profile", "subtracted", "profile_img"]]
        )
        self.img_ax = self.axes["cone"]
        self.prof_sub_ax = self.axes["subtracted"]
        self.profile_ax = self.axes["profile"]
        self.prof_img_ax = self.axes["profile_img"]
        # self.img_ax = self.axes[0]
        # self.prof_sub_ax = self.axes[1]
        # self.profile_ax = self.axes[2]
        # self.prof_img_ax = self.axes[3]
        self.img_ax.set_aspect("equal")  # type: ignore
        self.img_ax.set_title(
            f"Select radial profile for filter {str(self.filter_type)}"
        )

    def create_cone_slider_mpl_elements(self):
        # extract profiles in a cone around the selection from -angle to +angle from the profile selection vector
        self.profile_extraction_cone_size_radians = np.pi / 16
        # slider to select cone size
        self.cone_size_slider_ax = self.fig.add_axes([0.25, 0.05, 0.5, 0.03])  # type: ignore
        self.profile_extraction_cone_size_slider = Slider(
            ax=self.cone_size_slider_ax,
            label="cone size",
            valmin=0.0,
            valmax=np.pi,
            valinit=self.profile_extraction_cone_size_radians,
        )
        # slider hook
        self.profile_extraction_cone_size_slider.on_changed(self.update_cone_size)

    def create_image_mpl_elements(self):
        self.colormap = "magma"
        self.zscale = ZScaleInterval()
        self.img_vmin, self.img_vmax = self.zscale.get_limits(self.img)
        self.img_plot = self.img_ax.imshow(  # type: ignore
            self.img,
            vmin=self.img_vmin,
            vmax=self.img_vmax,
            origin="lower",
            cmap=self.colormap,
        )
        self.prof_sub_plot = self.prof_sub_ax.imshow(
            self.img,
            vmin=self.img_vmin,
            vmax=self.img_vmax,
            origin="lower",
            cmap=self.colormap,
        )
        self.prof_img_plot = self.prof_img_ax.imshow(
            self.img,
            vmin=self.img_vmin,
            vmax=self.img_vmax,
            origin="lower",
            cmap=self.colormap,
        )

    # TODO: constructor to initialize from a finished analysis
    # @classmethod
    # def from_saved_state(self, ...):

    def setup_mesh(self):
        """
        Create a distance-from-center mesh we can calculate once and store because the images do not change size
        """
        self.distance_from_center_mesh = calculate_distance_from_center_mesh(
            img=self.img
        )

    def onclick(self, event):
        # check that the click was in the image, and handle it if so
        if event.inaxes != self.img_ax:
            return
        rounded_x = int(np.round(event.xdata))
        rounded_y = int(np.round(event.ydata))
        self.profile_end = PixelCoord(x=rounded_x, y=rounded_y)
        self.update_plots()

    def update_cone_size(self, _):
        self.profile_extraction_cone_size_radians = (
            self.profile_extraction_cone_size_slider.val
        )
        self.update_plots()

    def update_plots(self):
        self.update_profile_extraction()
        self.update_profile_plot()
        self.update_profile_subtraction_plots()
        self.fig.canvas.draw_idle()  # type: ignore

    def create_profile_extraction_mpl_elements(self):
        self.extraction_line = plt.Line2D(  # type: ignore
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )

        # "left" edge of cone, each graph needs a separate line object
        self.cone_neg_line = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.img_ax.add_line(self.cone_neg_line)

        self.cone_pos_line = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )
        self.img_ax.add_line(self.cone_pos_line)
        self.img_ax.add_line(self.extraction_line)

    def update_profile_extraction(self):
        # given our clicked coordinate, figure out the radius and angle direction of the profile
        x0, y0 = self.image_center.x, self.image_center.y
        x1, y1 = self.profile_end.x, self.profile_end.y
        r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        r = int(np.round(r))

        # figure out the angles of the cone edges and the middle
        self.extraction_cone_mid_angle = np.arctan2(y1 - y0, x1 - x0)
        self.extraction_cone_min_angle = (
            self.extraction_cone_mid_angle - self.profile_extraction_cone_size_radians
        )
        self.extraction_cone_max_angle = (
            self.extraction_cone_mid_angle + self.profile_extraction_cone_size_radians
        )

        theta = self.extraction_cone_mid_angle
        cone_size = self.profile_extraction_cone_size_radians

        # for drawing lines from comet center to the edges of the extraction cone
        cone_neg_endpoint = PixelCoord(
            x=r * np.cos(theta - cone_size) + x0,
            y=r * np.sin(theta - cone_size) + y0,
        )
        cone_pos_endpoint = PixelCoord(
            x=r * np.cos(theta + cone_size) + x0,
            y=r * np.sin(theta + cone_size) + y0,
        )

        # get the median profiles in the cone
        self.radial_profile = extract_comet_radial_median_profile_from_cone(
            img=self.img,
            comet_center=self.image_center,
            r=r,
            theta=theta,
            cone_size=cone_size,
        )

        # update middle line of cone
        self.extraction_line.set_xdata([x1, x0])
        self.extraction_line.set_ydata([y1, y0])

        # update bounding lines of the cone
        self.cone_neg_line.set_xdata([cone_neg_endpoint.x, x0])
        self.cone_neg_line.set_ydata([cone_neg_endpoint.y, y0])
        self.cone_pos_line.set_xdata([cone_pos_endpoint.x, x0])
        self.cone_pos_line.set_ydata([cone_pos_endpoint.y, y0])

    def update_profile_plot(self):
        # have we already plotted a profile? clear it now
        if self.profile_plot is not None:
            self.profile_ax.clear()

        rs_in_km = (
            self.radial_profile.profile_axis_rs[1:] * self.epoch_summary.km_per_pix
        )
        self.profile_plot = self.profile_ax.plot(
            rs_in_km,
            # np.log10(uw1_pix_to_km),
            self.radial_profile.pixel_values[1:],
            # np.log10(self.uw1_radial_profile.pixel_values[1:]),
        )
        # draw horizontal shaded bars for 1, 2, and 3 sigma background levels: overlaying with alpha values will make lower sigmas darker
        for i in range(1, 4):
            self.profile_ax.axhspan(
                -i * self.bg_shot_noise_error,
                i * self.bg_shot_noise_error,
                color="blue",
                alpha=0.05,
            )

    def update_profile_subtraction_plots(self):
        img_copy = copy.deepcopy(self.img)
        self.profile_img = radial_profile_to_image(
            profile=self.radial_profile,
            distance_from_center_mesh=self.distance_from_center_mesh,
            empty_pixel_fill_value=0.0,
        )
        self.profile_sub_img = img_copy - self.profile_img
        self.prof_sub_plot.set_data(self.profile_sub_img)
        self.prof_img_plot.set_data(self.profile_img)

    def show(self):
        plt.show()


def profile_extraction_from_cone(
    scp: Products, ref: ProductReference
) -> CometRadialProfileFromConicalRegion:
    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)
    epoch_id = pkey.epoch_id

    eid = scp.load_epoch_index_entry(epoch_id=epoch_id)
    assert eid is not None

    img_ref = ProductReference(kind=ProductKind.bg_subtracted_stacked_image, key=pkey)
    img_fits = scp.load_fits_image(ref=img_ref)
    assert img_fits is not None
    assert isinstance(img_fits.data, SwiftUvotImage)

    bgr = scp.load_background_result(key=pkey)
    assert bgr is not None

    # TODO: check for extraction cone results from all filters and pass those in for display
    reference_profiles = None

    rpsp = RadialProfileSelectionPlot(
        eid=eid,
        img=img_fits.data,
        bgr=bgr,
        reference_profiles=reference_profiles,
        filter_type=pkey.filter_type,
    )
    rpsp.show()

    return rpsp.radial_profile
