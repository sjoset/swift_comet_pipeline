from dataclasses import dataclass

from rich.console import RenderResult
from rich.text import Text

from swift_comet_pipeline.pipeline.product_system.product_key import GlobalKey, KeyLike
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# References (combines kind & key to completely specify product)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductReference:
    kind: ProductKind
    key: KeyLike = GlobalKey()

    def __str__(self) -> str:
        keystr = str(self.key)
        return f"[{self.kind.value:<30}]: {keystr}"

    def __rich_console__(self, *_) -> RenderResult:
        yield Text(self.__str__(), end="")
