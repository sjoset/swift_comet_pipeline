from dataclasses import dataclass, field
from enum import IntFlag, auto
from pathlib import Path
from typing import Callable, Iterable

from swift_comet_pipeline.data_products.product_kind import ProductKind
from swift_comet_pipeline.data_products.product_ref import ProductRef


class KeyDim(IntFlag):
    """
    Describes whether a product varies with respect to the categories below
    """

    no_key = 0
    by_epoch = auto()
    by_filter = auto()
    by_stack_method = auto()


@dataclass(frozen=True)
class ProductFamily:
    """
    For each kind of product, stores how it changes across epoch, filter, etc.,
    and stores a method that determines the path this should be stored, when given a complete ProductRef.
    Also
    """

    kind: ProductKind
    dims: KeyDim
    locator: Callable[[ProductRef], Path]
    deps: Callable[[ProductRef], Iterable[ProductRef]]


# the dir
_folder_order = [KeyDim.by_epoch, KeyDim.by_filter, KeyDim.by_stack_method]


@dataclass(frozen=True)
class ProductPathScheme:

    stem_by_kind: dict[ProductKind, str] = field(default_factory=dict)
    ext_by_kind: dict[ProductKind, str] = field(default_factory=dict)
