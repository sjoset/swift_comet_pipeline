from __future__ import annotations

import os
import shutil

from pathlib import Path
from typing import Any, Callable, Dict, Protocol, Type

from swift_comet_pipeline.pipeline.product_system.product_key import KeyLike
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.pipeline.product_system.product_specification import (
    ProductSpecification,
)
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)


# -----------------------------------------------------------------------------
# Store interface (for IO backends) and a local filesystem implementation
# -----------------------------------------------------------------------------
class ProductStorage(Protocol):
    def exists(self, path: Path) -> bool: ...
    def atomic_write(self, tmp_file: Path, dest: Path) -> None: ...


class LocalFSProductStorage:
    def exists(self, path: Path) -> bool:
        return path.exists()

    def atomic_write(self, tmp_file: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_in_place = dest.with_suffix(dest.suffix + ".tmp")
        shutil.move(str(tmp_file), str(tmp_in_place))
        os.replace(tmp_in_place, dest)


# -----------------------------------------------------------------------------
# Subdirectory resolver: maps a data type to a subdirectory template
# -----------------------------------------------------------------------------
# A resolver may be a simple format string or a callable for complex logic.
KeySubdirFunc = Callable[[KeyLike, CometProjectConfig], Path]


class SubdirResolver:
    """
    Provides paths inside our project for a KeyLike, based on a template OR function
    Registering a template or function is mutually exclusive - we want a non-ambiguous path resolution
    """

    def __init__(self):
        self._formats: Dict[Type, str] = {}
        self._funcs: Dict[Type, KeySubdirFunc] = {}

    def register_template(self, key_type: Type, subdir_template: str) -> None:
        if key_type in self._funcs:
            print(
                f"SubdirResolver already has function to resolve keys of type {key_type}!"
            )
            print(f"Not registering the template {subdir_template}.")
            return
        self._formats[key_type] = subdir_template

    def register_func(self, key_type: Type, func: KeySubdirFunc) -> None:
        if key_type in self._formats:
            print(
                f"SubdirResolver already has function to resolve keys of type {key_type}!"
            )
            print(f"Not registering the function template.")
            return
        self._funcs[key_type] = func

    def resolve_relative_path(self, cfg: CometProjectConfig, key: KeyLike) -> Path:
        # Try callable first
        for cls in type(key).mro():
            fn = self._funcs.get(cls)
            if fn:
                return Path(fn(key, cfg))
        # Then try a template
        for cls in type(key).mro():
            tmpl = self._formats.get(cls)
            if tmpl:
                # Template can access cfg and key attributes directly
                return Path(tmpl.format(cfg=cfg, key=key))
        raise ValueError(f"No resolution for relative path of {key=}")


# -----------------------------------------------------------------------------
# Product spec & registry
# -----------------------------------------------------------------------------
class ProductRegistry:
    def __init__(
        self,
        store: ProductStorage | None = None,
        subdirs: SubdirResolver | None = None,
    ):
        self._specs: Dict[ProductReference, ProductSpecification] = {}
        self._store = store or LocalFSProductStorage()
        self._subdir_resolver = subdirs or SubdirResolver()

    # specs
    def register(self, spec: ProductSpecification) -> None:
        if spec.ref in self._specs:
            raise ValueError(f"ProductReference {spec.ref} is already registered")
        self._specs[spec.ref] = spec

    def spec(self, ref: ProductReference) -> ProductSpecification | None:
        # Return the ProductSpecification associated with the given ProductReference
        # try:
        #     return self._specs[ref]
        # except KeyError:
        #     raise KeyError(f"No ProductSpec registered for {ref}")
        return self._specs.get(ref, None)

    # Subdir resolution
    def subdir_resolver(self) -> SubdirResolver:
        return self._subdir_resolver

    # Uses the subdir resolver and the specification's template to construct the full path to product
    # Full path construction
    def path_for(self, ref: ProductReference, cfg: CometProjectConfig) -> Path | None:
        spec = self.spec(ref=ref)
        if spec is None:
            return None
        subdir = self._subdir_resolver.resolve_relative_path(cfg=cfg, key=ref.key)
        stem = spec.filename_stem_template.format(key=ref.key, cfg=cfg)
        return cfg.project_path / subdir / f"{stem}{spec.codec.suffix}"

    # IO
    def exists(self, ref: ProductReference, cfg: CometProjectConfig) -> bool:
        p = self.path_for(ref=ref, cfg=cfg)
        return False if p is None else self._store.exists(p)

    def save(
        self, ref: ProductReference, obj: Any, cfg: CometProjectConfig
    ) -> Path | None:
        spec = self.spec(ref=ref)
        dest = self.path_for(ref=ref, cfg=cfg)
        if spec is None or dest is None:
            return None
        tmp = dest.with_suffix(dest.suffix + ".writing")
        spec.codec.dump(obj, tmp)
        self._store.atomic_write(tmp, dest)
        return dest

    def load(self, ref: ProductReference, cfg: CometProjectConfig) -> Any:
        path = self.path_for(ref=ref, cfg=cfg)
        spec = self.spec(ref=ref)
        # print(f"Loading {spec} from {path}")
        if path is None or spec is None:
            return None
        if not path.exists():
            return None
        # if not path.exists():
        #     # raise FileNotFoundError(f"Missing product {ref.kind.name} at {path}")
        #     return None
        return spec.codec.load(path)

    def deps_for(self, ref: ProductReference) -> list[ProductReference]:
        spec = self.spec(ref=ref)
        if spec is None:
            return []
        if spec.deps is None:
            return []
        return list(spec.deps(ref))
