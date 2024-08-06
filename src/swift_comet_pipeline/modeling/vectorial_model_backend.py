from enum import StrEnum, auto

from icecream import ic


class VectorialModelBackend(StrEnum):
    sbpy = auto()
    rust = auto()
    # We don't include fortran here because pyvectorial_au does not support running it in parallel currently
    # fortran = auto()

    @classmethod
    def all_model_backends(cls):
        return [x for x in cls]


__VMBACKEND__: VectorialModelBackend | None = None


def vectorial_model_backend_init(backend: VectorialModelBackend) -> None:
    global __VMBACKEND__
    __VMBACKEND__ = backend


def get_vectorial_model_backend() -> VectorialModelBackend:
    global __VMBACKEND__

    if __VMBACKEND__ is None:
        ic(
            "Asked for vectorial model backend selection without it being configured! This is a bug!"
        )
        exit(1)

    return __VMBACKEND__
