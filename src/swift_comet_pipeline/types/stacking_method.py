from enum import StrEnum


class StackingMethod(StrEnum):
    summation = "sum"
    median = "median"

    @classmethod
    def all_stacking_methods(cls):
        return [x for x in cls]
