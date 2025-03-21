from icecream import ic

from swift_comet_pipeline.types.vectorial_model_backend import VectorialModelBackend


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
