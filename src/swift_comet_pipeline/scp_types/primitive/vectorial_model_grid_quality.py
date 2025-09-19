from enum import StrEnum, auto


class VectorialModelGridQuality(StrEnum):
    low = auto()
    medium = auto()
    high = auto()
    very_high = auto()

    @classmethod
    def all_qualities(cls):
        return [str(x) for x in cls]
