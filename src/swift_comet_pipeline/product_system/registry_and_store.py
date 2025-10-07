from __future__ import annotations

from functools import partial
from itertools import product
import json
import os
import shutil
from dataclasses import dataclass

# from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Protocol, Type

import pandas as pd
from astropy.io import fits
from astropy.table import Table

from swift_comet_pipeline.data_ingestion.observation_log.observation_log_io import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import (
    EpochIndex,
    epoch_index_from_json,
    json_from_epoch_index,
)
from swift_comet_pipeline.scp_types.compound.swift_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# Product kinds (kind chooses filename stem & codec; key chooses instance)
# -----------------------------------------------------------------------------
class ProductKind(StrEnum):
    # Ingestion / logs
    observation_log_raw = "raw observation log"
    observation_log_with_epochs = "observation log with epochs"
    observation_log_with_vetoes = "observation log with vetoes"
    orbit_data_earth = "earth orbit data"
    orbit_data_comet = "comet orbit data"
    epoch_index = "epoch index"

    # epoch subpipeline - for each epoch, filter, and stacking method, we have these products
    stacked_image_with_background = "stacked image, with bg"
    stacked_image_exposure_map = "stacked image exposure map"
    background_determination = "background level and error"
    bg_subtracted_stacked_image = "stacked image, no bg"
    # aperture_photometry_analysis = "variable-aperture photometry"
    # aperture_median_radial_profile = "median radial profile from apertures"
    # aperture_median_radial_profile_photometry = "photometry from median aperture radial profile"
    # radial_profile_from_cone = "radial profile from cone"
    # radial_profile_from_cone_photometry = "photometry from radial cone profile"

    # results assembly
    # afrho_from_apertures
    # afrho_from_aperture_median_profile
    # etc


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
class EpochSubpipelineKey(KeyLike):
    epoch_id: EpochID
    filter_type: UvotFilter
    stacking_method: StackingMethod

    def __str__(self):
        return f"{self.epoch_id}  {self.filter_type} {self.stacking_method}"


# -----------------------------------------------------------------------------
# References (combines kind & key to completely specify product)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductReference:
    kind: ProductKind
    key: KeyLike

    def __str__(self):
        keystr = f" {self.key}" if self.key != GlobalKey() else ""
        return f"[{self.kind.value:^30}]{keystr}"


# -----------------------------------------------------------------------------
# Codecs (format adapters with structural typing)
# -----------------------------------------------------------------------------
class Codec(Protocol):
    suffix: str

    def dump(self, obj: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...


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
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.writeto(path, overwrite=True)

    def load(self, path: Path) -> Any:
        hdul = fits.open(path, lazy_load_hdus=False, memmap=True)
        img = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img


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
        # TODO: cross-check for a registered function for key_type and throw exception
        self._formats[key_type] = subdir_template

    def register_func(self, key_type: Type, func: KeySubdirFunc) -> None:
        # TODO: cross-check for a registered template for key_type and throw exception
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
# Product specification: how do we name this kind of file and what are its deps?
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductSpecification:
    ref: ProductReference
    filename_stem_template: str
    codec: Codec
    deps: Callable[[ProductReference], Iterable[ProductReference]] | None


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

    # Specs
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


# -----------------------------------------------------------------------------
# Default registry with central key→subdir policy
# -----------------------------------------------------------------------------
def data_ingestion_registry() -> ProductRegistry:
    reg = ProductRegistry()

    reg.subdir_resolver().register_template(GlobalKey, ".")
    reg.subdir_resolver().register_template(
        EpochSubpipelineKey,
        "{key.epoch_id}/{key.filter_type}/{key.stacking_method}",
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            filename_stem_template="observation_log_raw",
            codec=ObservationLogCodec(),
            deps=None,
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
            ),
            filename_stem_template="observation_log_with_epochs",
            codec=ObservationLogCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey())
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
            ),
            filename_stem_template="observation_log_with_vetoes",
            codec=ObservationLogCodec(),
            deps=lambda _: [
                ProductReference(
                    kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
                )
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey()),
            filename_stem_template="orbital_data_earth",
            codec=PandasDataframeToCSVCodec(),
            deps=lambda _: [
                ProductReference(
                    kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
                )
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey()),
            filename_stem_template="orbital_data_comet",
            codec=PandasDataframeToCSVCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey())
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            filename_stem_template="epoch_index",
            codec=JSONCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey())
            ],
        )
    )

    return reg


def add_epoch_subpipelines_to_registry(
    reg: ProductRegistry, epoch_index: EpochIndex
) -> None:

    for epoch in epoch_index:

        epoch_id = epoch.epoch_id
        epoch_subpipe_key_func = partial(EpochSubpipelineKey, epoch_id=epoch_id)

        for filter_type, stacking_method in product(
            epoch.exposure_times.keys(), StackingMethod.all_stacking_methods()
        ):
            epoch_subpipe_key = epoch_subpipe_key_func(
                filter_type=filter_type, stacking_method=stacking_method
            )

            stacked_image_with_bg_ref = ProductReference(
                kind=ProductKind.stacked_image_with_background, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=stacked_image_with_bg_ref,
                    filename_stem_template="stack_with_background",
                    codec=InternalFITSImageCodec(),
                    deps=lambda _: [
                        ProductReference(kind=ProductKind.epoch_index, key=GlobalKey())
                    ],
                )
            )

            exp_map_ref = ProductReference(
                kind=ProductKind.stacked_image_exposure_map, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=exp_map_ref,
                    filename_stem_template="exposure_map",
                    codec=InternalFITSImageCodec(),
                    deps=lambda _: [
                        ProductReference(kind=ProductKind.epoch_index, key=GlobalKey())
                    ],
                )
            )

    return


# -----------------------------------------------------------------------------
# Convenience facade that binds a project config and loads products from it
# -----------------------------------------------------------------------------
class Products:
    """
    Facade to simplify dealing with product storage and retrieval
    This should be regenerated when products are deleted/added.
    """

    # TODO: rewrite function with registry_load/save

    def __init__(self, cfg: CometProjectConfig):
        self.cfg = cfg
        self._generate_registry()

    def _generate_registry(self):
        self.reg = data_ingestion_registry()
        self.epoch_index = self.load_epoch_index()

        if self.epoch_index is not None:
            add_epoch_subpipelines_to_registry(
                reg=self.reg, epoch_index=self.epoch_index
            )

        self.registry_load = partial(self.reg.load, cfg=self.cfg)
        self.registry_save = partial(self.reg.save, cfg=self.cfg)

    def regenerate(self):
        self._generate_registry()

    def exists(self, ref: ProductReference) -> bool:
        return self.reg.exists(ref=ref, cfg=self.cfg)

    def path_for(self, ref: ProductReference) -> Path | None:
        return self.reg.path_for(ref=ref, cfg=self.cfg)

    def load_raw_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.reg.load(
            ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            cfg=self.cfg,
        )

    def save_raw_log(self, df: SwiftUvotObservationLogDataframe) -> Path | None:
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    def load_epoch_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.reg.load(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
            ),
            cfg=self.cfg,
        )

    def save_epoch_log(self, df) -> Path | None:
        return self.reg.save(
            ref=ProductReference(ProductKind.observation_log_with_epochs, GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    def load_obs_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.reg.load(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
            ),
            cfg=self.cfg,
        )

    def save_obs_log(self, df) -> Path | None:
        return self.reg.save(
            ref=ProductReference(ProductKind.observation_log_with_vetoes, GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    def load_earth_orbit_data(self) -> pd.DataFrame | None:
        return self.reg.load(
            ref=ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey()),
            cfg=self.cfg,
        )

    def save_earth_orbit_data(self, df) -> Path | None:
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    def load_comet_orbit_data(self) -> pd.DataFrame | None:
        return self.reg.load(
            ref=ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey()),
            cfg=self.cfg,
        )

    def save_comet_orbit_data(self, df) -> Path | None:
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey()),
            obj=df,
            cfg=self.cfg,
        )

    # Epoch index
    def load_epoch_index(self):
        json_dict = self.reg.load(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            cfg=self.cfg,
        )
        if json_dict is None:
            return None
        return epoch_index_from_json(json_dict=json_dict)

    def save_epoch_index(self, epoch_index: EpochIndex) -> Path | None:
        json_dict = json_from_epoch_index(epoch_index=epoch_index)
        return self.reg.save(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            obj=json_dict,
            cfg=self.cfg,
        )

    def save_fits_image(self, img: fits.ImageHDU, ref: ProductReference):
        print(f"Saving {ref=}")
        return self.reg.save(ref=ref, obj=img, cfg=self.cfg)

    def load_fits_image(self, ref: ProductReference) -> fits.ImageHDU | None:
        return self.registry_load(ref=ref)
