from functools import partial
from itertools import product
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    WaterProductionKey,
)
from swift_comet_pipeline.pipeline.water_production_filter_pairs import (
    get_valid_water_production_filter_pairs,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.primitive import *

# TODO: rename this to product enumeration or something similar

# TODO: write function to search through directory for all dust rednesses that have been run, and build product references from that
# use the registry's name template or path_for with a dummy redness and then search the parent directory


# TODO: turn these into one function that takes a 'kind' parameter (or kinds=[...] like the rest of the arguments)
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


def enumerate_radial_profile_water_production_products(
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
                kind=ProductKind.radial_profile_water_production,
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


def enumerate_stacked_unbackgrounded_images(
    epochs: EpochIndex,
) -> list[ProductReference]:

    stackable_list = []
    for eid in epochs:
        available_keys = [
            EpochSubpipelineKey(epoch_id=eid.epoch_id, filter_type=f, stacking_method=s)
            for f, s in product(
                UvotFilter.all_filters(), StackingMethod.all_stacking_methods()
            )
            if eid.exposure_times.get(f, 0) > 0.0
        ]
        stackable_products = [
            ProductReference(kind=ProductKind.stacked_image_with_background, key=k)
            for k in available_keys
        ]
        stackable_list.extend(stackable_products)

    return stackable_list
