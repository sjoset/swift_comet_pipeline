from dataclasses import dataclass

from swift_comet_pipeline.data_products.product_key import ProductKeyLike
from swift_comet_pipeline.data_products.product_kind import ProductKind


@dataclass(frozen=True)
class ProductRef:
    kind: ProductKind
    key: ProductKeyLike
