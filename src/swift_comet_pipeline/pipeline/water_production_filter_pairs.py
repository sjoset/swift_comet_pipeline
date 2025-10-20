from itertools import product
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.compound.water_production_filter_pair import (
    WaterProductionFilterPair,
)
from swift_comet_pipeline.scp_types.primitive import *


def get_valid_water_production_filter_pairs(
    eid: EpochIndexEntry, oh_filters: list[UvotFilter], dust_filters: list[UvotFilter]
) -> list[WaterProductionFilterPair]:

    oh_and_dust_pairs = product(oh_filters, dust_filters)

    valid_pairs = [
        WaterProductionFilterPair(oh_filter=oh, dust_filter=dust)
        for oh, dust in oh_and_dust_pairs
        if eid.exposure_times.get(oh, 0) > 0 and eid.exposure_times.get(dust, 0) > 0
    ]

    return valid_pairs
