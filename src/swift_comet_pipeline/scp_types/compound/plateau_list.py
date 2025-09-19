from typing import TypeAlias

from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)
from swift_comet_pipeline.scp_types.primitive.plateau import ProductionPlateau


ReddeningToProductionPlateauListDict: TypeAlias = dict[
    DustReddeningPercent, list[ProductionPlateau]
]
