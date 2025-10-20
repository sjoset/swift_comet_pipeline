from itertools import product
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ProductKind,
    ProductReference,
    WaterProductionKey,
)
from swift_comet_pipeline.pipeline.water_production_filter_pairs import (
    get_valid_water_production_filter_pairs,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.primitive import *


def enumerate_aperture_water_production_products(
    epochs: EpochIndex,
    oh_filters: list[UvotFilter],
    dust_filters: list[UvotFilter],
    stacking_methods: list[StackingMethod],
    dust_rednesses: list[DustReddeningPercent],
) -> list[ProductReference]:

    terminal_products: list[ProductReference] = []
    for eid in epochs:
        filter_pairs = get_valid_water_production_filter_pairs(
            eid=eid, oh_filters=oh_filters, dust_filters=dust_filters
        )
        for fp, stacking_method, dust_redness in product(
            filter_pairs, stacking_methods, dust_rednesses
        ):
            pref = ProductReference(
                kind=ProductKind.aperture_water_production,
                key=WaterProductionKey(
                    epoch_id=eid.epoch_id,
                    oh_filter=fp.oh_filter,
                    dust_filter=fp.dust_filter,
                    stacking_method=stacking_method,
                    dust_redness_pct_per_hundred_nm=dust_redness,
                ),
            )
            terminal_products.append(pref)

    return terminal_products


# TODO: write function to search through directory for all dust rednesses that have been run, and build product references from that
# use the registry's name template or path_for with a dummy redness and then search the parent directory
