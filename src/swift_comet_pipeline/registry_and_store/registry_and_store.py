from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Type

import pandas as pd
from astropy.io import fits
from astropy.table import Table

from swift_comet_pipeline.data_ingestion.observation_log.observation_log_io import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.scp_types.compound.swift_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# Product kinds (kind chooses filename stem & codec; key chooses instance)
# -----------------------------------------------------------------------------
class ProductKind(Enum):
    # Ingestion / logs
    observation_log_raw = auto()
    observation_log_with_epochs = auto()
    epoch_index = auto()

    # epoch subpipeline
    # stacked_image_with_background = auto()

    # IMG_STACK_MEDIAN = auto()


# -----------------------------------------------------------------------------
# Keys (pure data; no path/formatting behavior)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyLike:
    pass


@dataclass(frozen=True)
class GlobalKey(KeyLike):
    pass


@dataclass(frozen=True)
class ProductKey(KeyLike):
    epoch_id: str
    filt: UvotFilter
    method: StackingMethod


# -----------------------------------------------------------------------------
# References (combines kind & key to completely specify product)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductReference:
    kind: ProductKind
    key: KeyLike


# -----------------------------------------------------------------------------
# Codecs (format adapters with structural typing)
# -----------------------------------------------------------------------------
class Codec(Protocol):
    suffix: str

    def dump(self, obj: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...


# TODO: csv, ecsv


class ObservationLogCodec:
    suffix = ".parquet"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_observation_log(obs_log=obj, obs_log_path=path)

    def load(self, path: Path) -> Any:
        return read_observation_log(obs_log_path=path)


class JSONCodec:
    suffix = ".json"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def load(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class PandasDataframeToCSVCodec:
    suffix = ".csv"

    def dump(self, obj: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)

    def load(self, path: Path) -> Any:
        return pd.read_csv(path)


class PandasDataframeToECSVCodec:
    suffix = ".ecsv"

    def dump(self, obj: Any, path: Path) -> None:
        t = Table.from_pandas(obj)
        t.meta = obj.attrs
        t.write(path, format="ascii.ecsv", overwrite=True)

    def load(self, path: Path) -> Any:
        t = Table.read(path, format="ascii.ecsv")
        df = t.to_pandas()
        df.attrs.update(t.meta)
        return df


class InternalFITSImageCodec:
    """
    We use only extension 1 to store image/header info for FITS we create ourselves
    """

    suffix = ".fits"

    def dump(self, obj: Any, path: Path) -> None:
        obj.writeto(path, overwrite=True)

    def load(self, path: Path) -> Any:
        hdul = fits.open(path, lazy_load_hdus=False, memmap=True)
        img = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img

    # def dump(self, obj: Any, path: Path) -> None:
    #     if fits is None:
    #         raise RuntimeError("astropy is required for FITSCodec")
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     if isinstance(obj, fits.HDUList):
    #         obj.writeto(path, overwrite=True)
    #     else:
    #         if np is None:
    #             raise RuntimeError("numpy is required to create FITS from array")
    #         hdu = fits.PrimaryHDU(np.asarray(obj))
    #         fits.HDUList([hdu]).writeto(path, overwrite=True)
    #
    # def load(self, path: Path) -> Any:
    #     if fits is None:
    #         raise RuntimeError("astropy is required for FITSCodec")
    #     return fits.open(path)


# class NPYCodec:
#     suffix = ".npy"
#
#     def dump(self, obj: Any, path: Path) -> None:
#         if np is None:
#             raise RuntimeError("numpy is required for NPYCodec")
#         path.parent.mkdir(parents=True, exist_ok=True)
#         np.save(path, obj)
#
#     def load(self, path: Path) -> Any:
#         if np is None:
#             raise RuntimeError("numpy is required for NPYCodec")
#         return np.load(path, allow_pickle=False)


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
    def __init__(self):
        self._formats: Dict[Type, str] = {}
        self._funcs: Dict[Type, KeySubdirFunc] = {}

    def register_template(self, key_type: Type, subdir_template: str) -> None:
        self._formats[key_type] = subdir_template

    def register_func(self, key_type: Type, func: KeySubdirFunc) -> None:
        self._funcs[key_type] = func

    # TODO: fix artifacts and default path
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
        # Fallback: project scope root
        return Path(f"artifacts/{cfg.comet_id}")


# -----------------------------------------------------------------------------
# Product specification: how do we name this kind of file and what are its deps?
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductSpecification:
    kind: ProductKind
    filename_stem_template: str
    codec: Codec
    deps: Callable[[ProductReference], Iterable[ProductReference]] | None


# TODO: fix function calls that dont use f(x=val, ...) form and just f(x, y, z)
# -----------------------------------------------------------------------------
# Product spec & registry
# -----------------------------------------------------------------------------
class ProductRegistry:
    def __init__(
        self,
        store: Optional[ProductStorage] = None,
        subdirs: Optional[SubdirResolver] = None,
    ):
        self._specs: Dict[ProductKind, ProductSpecification] = {}
        self._store = store or LocalFSProductStorage()
        self._subdir_resolver = subdirs or SubdirResolver()

    # Specs
    def register(self, spec: ProductSpecification) -> None:
        if spec.kind in self._specs:
            raise ValueError(f"ProductKind {spec.kind} is already registered")
        self._specs[spec.kind] = spec

    def spec(self, kind: ProductKind) -> ProductSpecification:
        try:
            return self._specs[kind]
        except KeyError:
            raise KeyError(f"No ProductSpec registered for {kind}")

    # Subdir resolution
    def subdir_resolver(self) -> SubdirResolver:
        return self._subdir_resolver

    # Full path construction
    def path_for(self, ref: ProductReference, cfg: CometProjectConfig) -> Path:
        spec = self.spec(ref.kind)
        subdir = self._subdir_resolver.resolve_relative_path(cfg=cfg, key=ref.key)
        stem = spec.filename_stem_template.format(key=ref.key, cfg=cfg)
        return cfg.project_path / subdir / f"{stem}{spec.codec.suffix}"

    # IO
    def exists(self, ref: ProductReference, cfg: CometProjectConfig) -> bool:
        return self._store.exists(self.path_for(ref=ref, cfg=cfg))

    def save(self, ref: ProductReference, obj: Any, cfg: CometProjectConfig) -> Path:
        spec = self.spec(kind=ref.kind)
        dest = self.path_for(ref=ref, cfg=cfg)
        tmp = dest.with_suffix(dest.suffix + ".writing")
        spec.codec.dump(obj, tmp)
        self._store.atomic_write(tmp, dest)
        return dest

    def load(self, ref: ProductReference, cfg: CometProjectConfig) -> Any:
        path = self.path_for(ref=ref, cfg=cfg)
        if not path.exists():
            raise FileNotFoundError(f"Missing product {ref.kind.name} at {path}")
        return self.spec(ref.kind).codec.load(path)


# -----------------------------------------------------------------------------
# Default registry with central key→subdir policy
# -----------------------------------------------------------------------------
# TODO: fix artifacts and default directory
def default_registry() -> ProductRegistry:
    reg = ProductRegistry()

    # Centralized subdir scheme (editable in one place)
    reg.subdir_resolver().register_template(GlobalKey, "artifacts/{cfg.comet_id}")
    reg.subdir_resolver().register_template(
        ProductKey,
        "artifacts/{cfg.comet_id}/stacks/{key.epoch_id}/{key.filt}/{key.method}",
    )

    # Product specs (kind → filename stem + codec)
    reg.register(
        ProductSpecification(
            kind=ProductKind.observation_log_raw,
            filename_stem_template="obs_log_raw",
            codec=ObservationLogCodec(),
            deps=None,
        )
    )

    reg.register(
        ProductSpecification(
            kind=ProductKind.observation_log_with_epochs,
            filename_stem_template="obs_log_with_epochs",
            codec=ObservationLogCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey())
            ],
        )
    )

    reg.register(
        ProductSpecification(
            kind=ProductKind.epoch_index,
            filename_stem_template="epoch_index",
            codec=PandasDataframeToCSVCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey())
            ],
        )
    )

    # # Downstream examples
    # reg.register(
    #     ProductSpecification(
    #         ProductKind.stacked_image_with_background,
    #         "stack_sum",
    #         FITS,
    #         deps=lambda key: [(ProductKind.observation_log_with_epochs, GlobalKey())],
    #     )
    # )
    # reg.register(
    #     ProductSpecification(
    #         ProductKind.IMG_STACK_MEDIAN,
    #         "stack_median",
    #         FITS,
    #         deps=lambda key: [(ProductKind.observation_log_with_epochs, GlobalKey())],
    #     )
    # )

    return reg


# TODO: fix function calls with f(x, y, z)
# -----------------------------------------------------------------------------
# Convenience facade that binds a project config and loads products from it
# -----------------------------------------------------------------------------
class Products:
    def __init__(self, cfg: CometProjectConfig, reg: Optional[ProductRegistry] = None):
        self.cfg = cfg
        self.reg = reg or default_registry()

    # Logs (global)
    def load_raw_log(self) -> SwiftUvotObservationLogDataframe:
        return self.reg.load(
            ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            cfg=self.cfg,
        )

    def save_raw_log(self, df: SwiftUvotObservationLogDataframe) -> Path:
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    def load_obs_log(self):
        return self.reg.load(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
            ),
            cfg=self.cfg,
        )

    def save_obs_log(self, df) -> Path:
        return self.reg.save(
            ref=ProductReference(ProductKind.observation_log_with_epochs, GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    # Epoch index (global in this scheme)
    def load_epoch_index(self):
        return self.reg.load(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            cfg=self.cfg,
        )

    def save_epoch_index(self, df) -> Path:
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    # # Stacks (examples)
    # def save_stack_sum(self, epoch_id: str, filt: str, arr_or_hdul) -> Path:
    #     return self.reg.save(
    #         ProductKind.stacked_image_with_background,
    #         EpochStackKey(epoch_id, filt, "sum"),
    #         arr_or_hdul,
    #         self.cfg,
    #     )
    #
    # def save_stack_median(self, epoch_id: str, filt: str, arr_or_hdul) -> Path:
    #     return self.reg.save(
    #         ProductKind.IMG_STACK_MEDIAN,
    #         EpochStackKey(epoch_id, filt, "median"),
    #         arr_or_hdul,
    #         self.cfg,
    #     )

    # Debug helper
    def path_for(self, ref: ProductReference) -> Path:
        return self.reg.path_for(ref=ref, cfg=self.cfg)


# -----------------------------------------------------------------------------
# Adapter for existing observation-log builder (unchanged API)
# -----------------------------------------------------------------------------


# def integrate_existing_obslog_builder(
#     cfg: ProjectConfig,
#     build_fn: Callable[[], Any],
#     *,
#     save_raw: bool = True,
# ) -> Path:
#     reg = default_registry()
#     df = build_fn()
#     if save_raw:
#         reg.save(ProductKind.observation_log_raw, GlobalKey(), df, cfg)
#     if pd is not None and "epoch_id" not in getattr(df, "columns", []):
#         df = df.copy()
#         df["epoch_id"] = pd.Series([None] * len(df), dtype="object")
#     return reg.save(ProductKind.observation_log_with_epochs, GlobalKey(), df, cfg)
