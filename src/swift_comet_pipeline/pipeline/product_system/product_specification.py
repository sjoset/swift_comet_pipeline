from dataclasses import dataclass
from typing import Callable, Iterable

from swift_comet_pipeline.pipeline.product_system.codecs import Codec
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)


# -----------------------------------------------------------------------------
# Product specification: how do we name this kind of file and what are its deps?
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductSpecification:
    # this product's ProductReference
    ref: ProductReference
    # stem of the file name (without the extension)
    filename_stem_template: str
    codec: Codec
    deps: Callable[[ProductReference], Iterable[ProductReference]] | None
