from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive import *


@dataclass(frozen=True)
class ProductKey:
    epoch_id: str
    filter: str
    stacking_method: StackingMethod


@dataclass(frozen=True)
class GlobalKey(ProductKey):
    global_key = "global"


ProductKeyLike: TypeAlias = ProductKey | GlobalKey
