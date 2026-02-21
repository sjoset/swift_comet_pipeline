from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive import *


@dataclass(frozen=True)
class WaterProductionFilterPair:
    oh_filter: UvotFilter
    dust_filter: UvotFilter
