from typing import TypeAlias

from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.plateau import ProductionPlateau


ReddeningToProductionPlateauListDict: TypeAlias = dict[
    DustReddeningPercent, list[ProductionPlateau]
]
