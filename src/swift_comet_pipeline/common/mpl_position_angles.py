from matplotlib.patches import FancyArrowPatch
import numpy as np

from swift_comet_pipeline.scp_types.primitive import *


def add_position_angles(
    ax,
    at_coords_fraction: tuple[float, float],
    position_angles: TailPositionAngles,
    # size=0.1,
    v_arrow_size: float = 0.1,
    sun_arrow_size: float = 0.1,
    sun_arrow_x_offset: float = 0.0,
    sun_arrow_y_offset: float = 0.0,
    v_text_x_offset: float = 0,
    v_text_y_offset: float = 0,
    sun_text_x_offset: float = 0,
    sun_text_y_offset: float = 0,
    v_arrow_alpha: float = 1.0,
    sun_arrow_alpha: float = 1.0,
):

    x0, y0 = at_coords_fraction
    velocity_pa = position_angles.dust_tail_pa + (180 * u.deg)  # type: ignore
    sun_pa = position_angles.ion_tail_pa + (180 * u.deg)  # type: ignore

    v_kw = dict(arrowstyle="-|>", mutation_scale=10, alpha=v_arrow_alpha)
    sun_kw = dict(arrowstyle="-|>", mutation_scale=10, alpha=sun_arrow_alpha)

    # position angles to angles in our image
    v_angle_rad = np.deg2rad(-1 * velocity_pa.to_value(u.deg))  # type: ignore
    sun_angle_rad = np.deg2rad(-1 * sun_pa.to_value(u.deg))  # type: ignore

    # sunward_angle_rad = sun_angle_rad + np.pi
    sunward_angle_rad = sun_angle_rad

    sun_size_scale = 1.5
    v_size_scale = 1.2

    # tail vector components
    v_x_tail = x0 + v_size_scale * v_arrow_size * np.sin(v_angle_rad)
    v_y_tail = y0 + v_size_scale * v_arrow_size * np.cos(v_angle_rad)
    sun_x_tail = x0 + sun_size_scale * sun_arrow_size * np.sin(sunward_angle_rad)
    sun_y_tail = y0 + sun_size_scale * sun_arrow_size * np.cos(sunward_angle_rad)

    # dust
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (v_x_tail, v_y_tail),
            transform=ax.transAxes,
            color="#e7e7ea",
            **v_kw,  # type: ignore
        )
    )
    ax.text(
        v_x_tail + v_text_x_offset,
        v_y_tail + v_text_y_offset,
        r"$\vec{v}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#e7e7ea",
        alpha=0.9,
    )

    sun_hex_color = "#ffde21"
    # sun_hex_color = "#f9d71c"
    # sun_hex_color = "#ffe87c"
    # sun_hex_color = "#ffbf00"
    # sun_hex_color = "#ffea00"

    # sun
    ax.add_patch(
        FancyArrowPatch(
            (x0 + sun_arrow_x_offset, y0 + sun_arrow_y_offset),
            (sun_x_tail + sun_arrow_x_offset, sun_y_tail + sun_arrow_y_offset),
            transform=ax.transAxes,
            color=sun_hex_color,
            **sun_kw,  # type: ignore
        )
    )
    ax.text(
        sun_x_tail + sun_text_x_offset,
        sun_y_tail + sun_text_y_offset,
        r"$\odot$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=sun_hex_color,
        alpha=0.9,
    )
