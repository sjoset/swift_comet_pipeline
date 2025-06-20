from enum import StrEnum, auto


class VectorialFitType(StrEnum):
    near_fit = auto()
    far_fit = auto()
    full_fit = auto()

    @classmethod
    def all_types(cls):
        return [x for x in cls]
