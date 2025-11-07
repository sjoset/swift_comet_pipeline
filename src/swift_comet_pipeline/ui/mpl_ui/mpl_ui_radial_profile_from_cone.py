import copy
from typing import Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from astropy.visualization import ZScaleInterval

from swift_comet_pipeline.common.mpl_compass import add_compass
from swift_comet_pipeline.common.mpl_position_angles import add_position_angles
from swift_comet_pipeline.data_ingestion.orbit_data.position_angles import (
    get_position_angles,
)
from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.photometry.comet.extract_comet_radial_profile import (
    calculate_distance_from_center_mesh,
    extract_comet_radial_median_profile_from_cone,
    radial_profile_to_image,
)
from swift_comet_pipeline.pipeline.product_enumeration import enumerate_all_products_of
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


@dataclass
class MPLImageExtractionCone:
    # these will hold plt.Line2D artists
    mid_line_artist: Any
    left_edge_artist: Any
    right_edge_artist: Any

    # angular deviation from the central profile, half of the total cone size
    extraction_cone_angle_size_rad: float

    profile_begin: PixelCoord
    profile_end: PixelCoord

    @property
    def profile_radius_int(self) -> int:
        x0, y0 = (
            self.profile_begin.x,
            self.profile_begin.y,
        )
        x1, y1 = self.profile_end.x, self.profile_end.y
        return int(np.round(np.hypot(x1 - x0, y1 - y0)))

    @property
    def mid_angle_rad(self) -> float:
        x0, y0 = (
            self.profile_begin.x,
            self.profile_begin.y,
        )
        x1, y1 = self.profile_end.x, self.profile_end.y
        return np.arctan2(y1 - y0, x1 - x0)

    @property
    def left_angle_rad(self) -> float:
        return self.mid_angle_rad - self.extraction_cone_angle_size_rad

    @property
    def right_angle_rad(self) -> float:
        return self.mid_angle_rad + self.extraction_cone_angle_size_rad

    def update_artists(self) -> None:
        x0, y0 = (
            self.profile_begin.x,
            self.profile_begin.y,
        )
        x1, y1 = self.profile_end.x, self.profile_end.y

        r = self.profile_radius_int
        theta = self.mid_angle_rad
        cone_size = self.extraction_cone_angle_size_rad

        cone_neg_endpoint = PixelCoord(
            x=r * np.cos(theta - cone_size) + x0,
            y=r * np.sin(theta - cone_size) + y0,
        )
        cone_pos_endpoint = PixelCoord(
            x=r * np.cos(theta + cone_size) + x0,
            y=r * np.sin(theta + cone_size) + y0,
        )

        # update middle line of cone
        self.mid_line_artist.set_xdata([x1, x0])
        self.mid_line_artist.set_ydata([y1, y0])

        # update bounding lines of the cone
        self.left_edge_artist.set_xdata([cone_neg_endpoint.x, x0])
        self.left_edge_artist.set_ydata([cone_neg_endpoint.y, y0])
        self.right_edge_artist.set_xdata([cone_pos_endpoint.x, x0])
        self.right_edge_artist.set_ydata([cone_pos_endpoint.y, y0])


class RadialProfileExtractionPlot(object):
    # TODO: make this take img_axis and profile_axis as arguments to embed in other figures
    # TODO: draw reference profiles at low alpha after showing image
    # TODO: add buttons/key to increase radius without changing angle
    # TODO: use horizons id to set title
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
        jpl_horizons_id: str | None = None,
    ):
        self.eid = eid
        self.img = img
        self.filter_type = filter_type
        self.bgr = bgr
        self.bg_shot_noise_error = np.sqrt(self.bgr.bg_shot_noise_variance)

        self.jpl_horizons_id = jpl_horizons_id or "Unknown Comet"
        if self.jpl_horizons_id:
            self.position_angles = get_position_angles(
                jpl_horizons_id=self.jpl_horizons_id, at_time=self.eid.observation_time
            )
        else:
            self.position_angles = None

        self.reference_profiles = reference_profiles or {}
        self.reference_profile_index = 0

        self.image_center = get_uvot_image_center(self.img)
        self.time_from_perihelion = eid.time_from_perihelion

        # figure, axes for each graph
        self.create_basic_mpl_elements()

        # for displaying the stacked image
        self.create_image_mpl_elements()

        self.profile_plot = None

        # for the lines showing the extraction cone
        self.create_profile_extraction_mpl_elements()

        # for displaying the slider to adjust size of cone
        self.create_cone_slider_mpl_elements()

        # for quickly extracting radial profiles
        self.setup_mesh()

        # show position angles for sun/velocity
        self.create_compass_and_pa_mpl_elements()

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)  # type: ignore
        self.update_plots()

    def create_basic_mpl_elements(self):
        self.fig, self.axes = plt.subplot_mosaic(
            [["cone", "cone", "cone"], ["profile", "subtracted", "profile_img"]]
        )
        self.img_ax = self.axes["cone"]
        self.prof_sub_ax = self.axes["subtracted"]
        self.profile_ax = self.axes["profile"]
        self.prof_img_ax = self.axes["profile_img"]
        self.img_ax.set_aspect("equal")  # type: ignore
        self.img_ax.set_title(
            f"Select radial profile for filter {str(self.filter_type)}"
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def create_cone_slider_mpl_elements(self):
        # extract profiles in a cone around the selection from -angle to +angle from the profile selection vector
        # self.profile_extraction_cone_size_radians = np.pi / 16
        # slider to select cone size
        self.cone_size_slider_ax = self.fig.add_axes([0.25, 0.05, 0.5, 0.03])  # type: ignore
        self.profile_extraction_cone_size_slider = Slider(
            ax=self.cone_size_slider_ax,
            label="cone size",
            valmin=0.0,
            valmax=np.pi,
            # valinit=self.profile_extraction_cone_size_radians,
            valinit=self.extraction_cone.extraction_cone_angle_size_rad,
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
        self.extraction_cone.profile_end = PixelCoord(x=rounded_x, y=rounded_y)
        self.update_plots()

    def update_cone_size(self, _):
        # self.profile_extraction_cone_size_radians = (
        #     self.profile_extraction_cone_size_slider.val
        # )
        self.extraction_cone.extraction_cone_angle_size_rad = (
            self.profile_extraction_cone_size_slider.val
        )
        self.update_plots()

    def update_plots(self):
        self.update_profile_extraction()
        self.update_profile_plot()
        self.update_profile_subtraction_plots()
        self.fig.canvas.draw_idle()  # type: ignore

    def create_profile_extraction_mpl_elements(self):
        # initialize an arbitrary initial profile

        extraction_line = plt.Line2D(  # type: ignore
            xdata=[0, 0], ydata=[0, 0], lw=1, color="white", alpha=0.3
        )

        # "left" edge of cone
        cone_neg_line = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )

        # "right" edge of cone
        cone_pos_line = plt.Line2D(  # type: ignore
            xdata=[0, 0],
            ydata=[0, 0],
            lw=1,
            color="black",
            alpha=0.2,
        )

        self.extraction_cone = MPLImageExtractionCone(
            mid_line_artist=extraction_line,
            left_edge_artist=cone_neg_line,
            right_edge_artist=cone_pos_line,
            profile_begin=self.image_center,
            profile_end=PixelCoord(
                x=self.image_center.x + 50, y=self.image_center.y + 50
            ),
            extraction_cone_angle_size_rad=np.pi / 16,
        )

        self.img_ax.add_line(self.extraction_cone.mid_line_artist)
        self.img_ax.add_line(self.extraction_cone.left_edge_artist)
        self.img_ax.add_line(self.extraction_cone.right_edge_artist)

    def update_profile_extraction(self):

        # get the median profiles in the cone
        self.radial_profile = extract_comet_radial_median_profile_from_cone(
            img=self.img,
            comet_center=self.extraction_cone.profile_begin,
            r=self.extraction_cone.profile_radius_int,
            theta=self.extraction_cone.mid_angle_rad,
            cone_size=self.extraction_cone.extraction_cone_angle_size_rad,
        )

        self.extraction_cone.update_artists()

    def update_profile_plot(self):
        # have we already plotted a profile? clear it now
        if self.profile_plot is not None:
            self.profile_ax.clear()

        rs_in_km = self.radial_profile.profile_axis_rs[1:] * self.eid.km_per_pix
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

    def change_keymaps(self) -> None:
        # replace default h/l keybinds, but leave other binds for home/yscale untouched
        self.original_home_keymaps = mpl.rcParams["keymap.home"]
        self.original_yscale_keymaps = mpl.rcParams["keymap.yscale"]
        mpl.rcParams["keymap.home"] = [
            k for k in mpl.rcParams["keymap.home"] if k != "h"
        ]
        mpl.rcParams["keymap.yscale"] = [
            k for k in mpl.rcParams["keymap.yscale"] if k != "l"
        ]

    def restore_keymaps(self) -> None:
        mpl.rcParams["keymap.home"] = self.original_home_keymaps
        mpl.rcParams["keymap.yscale"] = self.original_yscale_keymaps

    def show(self):
        self.change_keymaps()
        plt.show()
        self.restore_keymaps()

    def on_key_press(self, event):
        if not self.reference_profiles:
            print("no reference profiles found")
            return
        if event.key == "l":
            self.snap_to_reference_profile(self.reference_profile_index + 1)
        elif event.key == "h":
            self.snap_to_reference_profile(self.reference_profile_index - 1)
        self.update_plots()

    def snap_to_reference_profile(self, new_reference_profile_index: int) -> None:
        reference_profile_keys = list(self.reference_profiles.keys())
        num_reference_profiles = len(reference_profile_keys)
        bounded_index = new_reference_profile_index % num_reference_profiles
        if bounded_index < 0:
            bounded_index += num_reference_profiles
        print(
            f"Changing reference profile index from {self.reference_profile_index} to {bounded_index}"
        )
        self.reference_profile_index = bounded_index

        ref_prof = list(self.reference_profiles.values())[bounded_index]
        new_endpoint = PixelCoord(x=ref_prof._xs[-1], y=ref_prof._ys[-1])
        new_cone_size = ref_prof._cone_size
        self.extraction_cone.profile_end = new_endpoint
        self.extraction_cone.extraction_cone_angle_size_rad = new_cone_size

    def draw_reference_profiles(self):
        # TODO: label each profile with text based on filter as well
        pass

    def create_compass_and_pa_mpl_elements(self):
        if not self.position_angles:
            return
        coords_fraction = (0.15, 0.85)
        add_position_angles(
            ax=self.img_ax,
            at_coords_fraction=coords_fraction,
            position_angles=self.position_angles,
            v_text_y_offset=0.03,
        )
        add_compass(
            ax=self.img_ax,
            at_coords_fraction=coords_fraction,
            north_arrow_color="#688894",
            north_text_color="#688894",
            east_arrow_color="#a4b7be",
            east_text_color="#a4b7be",
        )


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

    # check for extraction cone results from all filters and pass those in for display
    reference_profiles = {}

    # loop through all filters in the same epoch and load profiles if they exist
    other_profiles = enumerate_all_products_of(
        kind=ProductKind.radial_profile_from_cone,
        epochs=[eid],
        oh_filters=UvotFilter.all_filters(),
        dust_filters=UvotFilter.all_filters(),
        stacking_methods=[pkey.stacking_method],
    )

    for p in other_profiles:
        assert isinstance(p.key, EpochSubpipelineKey)
        print(f"Searching for reference profile {p.kind} --> {p.key}")
        if scp.exists(p):
            print(f"Reference profile found {p.kind} --> {p.key}")
            reference_profiles[p.key.filter_type] = scp.load_extracted_radial_profile(
                p.key
            )

    if not reference_profiles:
        reference_profiles = None

    rpsp = RadialProfileExtractionPlot(
        eid=eid,
        img=img_fits.data,
        bgr=bgr,
        reference_profiles=reference_profiles,
        filter_type=pkey.filter_type,
        jpl_horizons_id=scp.cfg.jpl_horizons_id,
    )
    rpsp.show()

    return rpsp.radial_profile
