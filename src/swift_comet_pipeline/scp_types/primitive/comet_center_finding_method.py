from enum import StrEnum, auto


class CometCenterFindingMethod(StrEnum):
    pixel_center = auto()
    aperture_centroid = auto()
    aperture_peak = auto()

    @classmethod
    def all_methods(cls):
        return [x for x in cls]
