import sys
import pathlib

from icecream import ic
from pyvectorial_au.db.cache_init import initialize_vectorial_model_cache

# from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


__VMCACHE_PATH__: pathlib.Path | None = None


def _construct_vectorial_model_cache_path(
    project_path: pathlib.Path,
) -> pathlib.Path:
    vmcache_path = (
        project_path
        / pathlib.Path("cache")
        / pathlib.Path("vectorial_model_cache.sqlite3")
    )

    return vmcache_path


def vectorial_model_cache_init(project_path: pathlib.Path) -> None:
    global __VMCACHE_PATH__
    if __VMCACHE_PATH__ is not None:
        return

    __VMCACHE_PATH__ = _construct_vectorial_model_cache_path(project_path=project_path)

    __VMCACHE_PATH__.parent.mkdir(parents=True, exist_ok=True)

    initialize_vectorial_model_cache(
        vectorial_model_cache_dir=__VMCACHE_PATH__.parent,
        vectorial_model_cache_filename=pathlib.Path(__VMCACHE_PATH__.name),
    )


def get_vectorial_model_cache_path() -> pathlib.Path:
    global __VMCACHE_PATH__
    if __VMCACHE_PATH__ is None:
        ic(
            "Asked for vectorial model cache path without initializing it first! This is a bug!"
        )
        sys.exit(1)

    return __VMCACHE_PATH__
