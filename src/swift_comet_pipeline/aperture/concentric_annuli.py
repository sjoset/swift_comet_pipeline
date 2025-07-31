import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture
from swift_comet_pipeline.types.pixel_coord import PixelCoord


def make_concentric_annular_apertures(
    ap_center: PixelCoord, min_radius: float, max_radius: float, num_slices: int
) -> list[CircularAnnulus | CircularAperture]:

    ap_position = (ap_center.x, ap_center.y)
    ap_edges = np.linspace(min_radius, max_radius, num=num_slices + 1, endpoint=True)

    # make the innermost aperture a circular one
    if min_radius == 0.0:
        # runs from ap_edge 0 to ap_edge 1
        circular_inner = CircularAperture(
            positions=(ap_center.x, ap_center.y), r=ap_edges[1]
        )

        annular_inners = ap_edges[1:-1]
        annular_outers = ap_edges[2:]
    else:
        annular_inners = ap_edges[:-1]
        annular_outers = ap_edges[1:]

    annular_list: list[CircularAnnulus | CircularAperture] = [
        CircularAnnulus(positions=ap_position, r_in=r_in, r_out=r_out)
        for r_in, r_out in zip(annular_inners, annular_outers)
    ]

    if min_radius == 0.0:
        return [circular_inner] + annular_list
    else:
        return annular_list
