from enum import StrEnum, auto


class BackgroundDeterminationMethod(StrEnum):
    swift_constant = auto()
    manual_aperture_mean = auto()
    manual_aperture_median = auto()
    gui_manual_aperture = auto()
    walking_aperture_ensemble = auto()
