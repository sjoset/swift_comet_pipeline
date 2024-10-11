from enum import StrEnum, auto


class BackgroundDeterminationMethod(StrEnum):
    # manual_aperture_mean = auto()
    # manual_aperture_median = auto()
    gui_manual_aperture = auto()
    swift_constant = auto()
    walking_aperture_ensemble = auto()

    @classmethod
    def all_bg_determination_methods(cls):
        return [x for x in cls]
