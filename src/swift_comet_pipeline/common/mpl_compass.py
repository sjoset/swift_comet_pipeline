from matplotlib.patches import FancyArrowPatch


# TODO: move these into one function


def add_compass(
    ax,
    at_coords_fraction: tuple[float, float],
    size=0.1,
    north_arrow_color: str = "white",
    east_arrow_color: str = "white",
    north_text_color: str = "white",
    east_text_color: str = "white",
):
    """
    Draw N/E arrows at fraction‐coords xy with length=size.
    kwargs passed to FancyArrowPatch (e.g. color, linewidth).
    """

    x0, y0 = at_coords_fraction
    kw = dict(arrowstyle="-|>", mutation_scale=10, alpha=0.9)

    # north
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x0, y0 + size), transform=ax.transAxes, color=north_arrow_color, **kw)  # type: ignore
    )
    ax.text(
        x0,
        y0 + size + 0.02,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=north_text_color,
        alpha=0.9,
    )

    # east
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x0 - size, y0), transform=ax.transAxes, color=east_arrow_color, **kw)  # type: ignore
        # FancyArrowPatch((x0, y0), (x0 - size, y0), transform=ax.transAxes, **kw)  # type: ignore
    )
    ax.text(
        # x0 - size / 2 - 0.01,
        x0 - size - 0.02,
        y0,
        "E",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=east_text_color,
        alpha=0.9,
    )
