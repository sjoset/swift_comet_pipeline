from matplotlib.lines import Line2D
from matplotlib.patches import Arc

from swift_comet_pipeline.scp_types.primitive import *


def draw_extraction_cone(
    ax,
    comet_center: PixelCoord,
    cone_length: float,
    cone_angle_rad: float,
    cone_size_rad: float,
) -> None:

    left_edge_rad = cone_angle_rad - cone_size_rad
    right_edge_rad = cone_angle_rad + cone_size_rad

    mid_end_x = comet_center.x + cone_length * np.cos(cone_angle_rad)
    mid_end_y = comet_center.y + cone_length * np.sin(cone_angle_rad)
    mid_extraction_line = Line2D(
        xdata=[comet_center.x, mid_end_x],
        ydata=[comet_center.y, mid_end_y],
        lw=2,
        color="white",
        alpha=0.3,
    )

    left_end_x = comet_center.x + cone_length * np.cos(left_edge_rad)
    left_end_y = comet_center.y + cone_length * np.sin(left_edge_rad)
    # left edge of cone
    left_edge_line = Line2D(
        xdata=[comet_center.x, left_end_x],
        ydata=[comet_center.y, left_end_y],
        lw=2,
        color="black",
        alpha=0.2,
    )
    ax.add_line(left_edge_line)

    right_end_x = comet_center.x + cone_length * np.cos(right_edge_rad)
    right_end_y = comet_center.y + cone_length * np.sin(right_edge_rad)
    # right edge
    right_edge_line = Line2D(
        xdata=[comet_center.x, right_end_x],
        ydata=[comet_center.y, right_end_y],
        lw=2,
        color="black",
        alpha=0.2,
    )
    ax.add_line(right_edge_line)

    ax.add_line(mid_extraction_line)

    arc = Arc(
        (comet_center.x, comet_center.y),
        cone_length,
        cone_length,
        angle=0,
        theta1=np.rad2deg(left_edge_rad),
        theta2=np.rad2deg(right_edge_rad),
        lw=2,
        edgecolor="black",
        alpha=0.2,
        linestyle="--",
    )
    ax.add_patch(arc)
