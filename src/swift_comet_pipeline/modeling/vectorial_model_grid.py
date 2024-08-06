from enum import StrEnum, auto

from icecream import ic
from pyvectorial_au.model_input.vectorial_model_config import VectorialModelGrid


class VectorialModelGridQuality(StrEnum):
    low = auto()
    medium = auto()
    high = auto()
    very_high = auto()

    @classmethod
    def all_qualities(cls):
        return [str(x) for x in cls]


__VMGRID_QUALITY__: VectorialModelGridQuality | None = None


def vectorial_model_grid_quality_init(quality: VectorialModelGridQuality) -> None:
    global __VMGRID_QUALITY__
    __VMGRID_QUALITY__ = quality


def make_vectorial_model_grid() -> VectorialModelGrid:
    global __VMGRID_QUALITY__

    if __VMGRID_QUALITY__ is None:
        ic("Asked for vectorial model grid without specifying quality! This is a bug!")
        exit(1)

    if __VMGRID_QUALITY__ == VectorialModelGridQuality.low:
        radial_points = 50
        angular_points = 30
        radial_substeps = 50
    elif __VMGRID_QUALITY__ == VectorialModelGridQuality.medium:
        radial_points = 100
        angular_points = 60
        radial_substeps = 75
    elif __VMGRID_QUALITY__ == VectorialModelGridQuality.high:
        radial_points = 150
        angular_points = 100
        radial_substeps = 100
    elif __VMGRID_QUALITY__ == VectorialModelGridQuality.very_high:
        radial_points = 200
        angular_points = 120
        radial_substeps = 120

    return VectorialModelGrid(
        radial_points=radial_points,
        angular_points=angular_points,
        radial_substeps=radial_substeps,
        parent_destruction_level=0.99,
        fragment_destruction_level=0.95,
    )
