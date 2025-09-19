from enum import StrEnum, auto


class VectorialModelBackend(StrEnum):
    sbpy = auto()
    rust = auto()
    # We don't include fortran here because pyvectorial_au does not support running it in parallel currently
    # fortran = auto()

    @classmethod
    def all_model_backends(cls):
        return [x for x in cls]
