from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from swift_comet_pipeline.data_products.product_kind import ProductKind


class GetProductError(Exception):
    kind: str


# TODO: clean up/expand as we go


@dataclass(frozen=True)
class Underspecified(GetProductError):
    kind: str = "Underspecified"
    message: str = ""


@dataclass(frozen=True)
class InvalidCombo(GetProductError):
    kind: str = "InvalidCombo"
    message: str = ""


@dataclass(frozen=True)
class Missing(GetProductError):
    kind: str = "Missing"
    reason: str = ""


@dataclass(frozen=True)
class Stale(GetProductError):
    kind: str = "Stale"
    message: str = ""
    dep: ProductArtifact | None = None


@dataclass(frozen=True)
class ProductOk:
    value: ProductArtifact


@dataclass(frozen=True)
class ProductErr:
    error: GetProductError


@dataclass(frozen=True)
class ReadError(GetProductError):
    kind: str = "ReadError"
    message: str = ""
    path: Path | None = None
    cause: BaseException | None = None
    product_kind: ProductKind | None = None


Result: TypeAlias = ProductOk | ProductErr
