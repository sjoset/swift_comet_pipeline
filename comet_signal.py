# import numpy as np
from photutils.aperture import (
    CircularAperture,
    ApertureStats,
    # CircularAnnulus,
)

# from dataclasses import dataclass
from typing import TypeAlias
from enum import Enum, auto


from swift_types import (
    # StackingMethod,
    # SwiftFilter,
    # filter_to_string,
    SwiftUVOTImage,
    # SwiftStackedUVOTImage,
)

__version__ = "0.0.1"


CometCountRate: TypeAlias = float


class CometPhotometryMethod(str, Enum):
    manual_aperture = auto()
    # auto_aperture = auto()


def comet_photometry(
    img: SwiftUVOTImage,
    photometry_method: CometPhotometryMethod,
    **kwargs,
) -> CometCountRate:
    if photometry_method == CometPhotometryMethod.manual_aperture:
        return comet_manual_aperture(img=img, **kwargs)


def comet_manual_aperture(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
) -> CometCountRate:
    # center_row = int(np.floor(img.shape[0] / 2))
    # center_col = int(np.floor(img.shape[1] / 2))
    # print(
    #     f"Using test aperture at image center: {center_col}, {center_row}"
    # )

    # # reminder: x coords are columns, y rows
    # initial_comet_aperture = CircularAperture(
    #     (center_col, center_row), r=comet_aperture_radius
    # )
    #
    # # use an aperture on the image center of the specified radius and find the centroid of the comet signal
    # initial_aperture_stats = ApertureStats(image_sum, initial_comet_aperture)
    # print(
    #     f"Moving analysis aperture to the centroid of the test aperture: {initial_aperture_stats.centroid}"
    # )

    # # Move the aperture to the centroid of the test aperture and do our analysis there
    # comet_center_x, comet_center_y = (
    #     initial_aperture_stats.centroid[0],
    #     initial_aperture_stats.centroid[1],
    # )
    comet_aperture = CircularAperture((aperture_x, aperture_y), r=aperture_radius)
    comet_aperture_stats = ApertureStats(img, comet_aperture)
    # print(f"Centroid of analysis aperture: {comet_aperture_stats.centroid}")

    comet_count_rate = float(comet_aperture_stats.sum)
    # # the sum images are in count rates, so multiply by exposure time for counts
    # comet_counts = comet_count_rate * exposure_time

    return comet_count_rate
