from itertools import product
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    ContinuumSubtractionKey,
)
from swift_comet_pipeline.pipeline.water_production_filter_pairs import (
    get_valid_water_production_filter_pairs,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.primitive import *

# TODO: write function to search through directory for all dust rednesses that have been run, and build product references from that
# use the registry's name template or path_for with a dummy redness and then search the parent directory


# TODO: remove old code
# def _enumerate_aperture_water_production_products(
#     epochs: EpochIndex,
#     oh_filters: list[UvotFilter],
#     dust_filters: list[UvotFilter],
#     stacking_methods: list[StackingMethod],
#     dust_rednesses: list[DustReddeningPercent],
# ) -> list[ProductReference]:
#
#     terminal_products: list[ProductReference] = []
#     for eid in epochs:
#         filter_pairs = get_valid_water_production_filter_pairs(
#             eid=eid, oh_filters=oh_filters, dust_filters=dust_filters
#         )
#         for fp, stacking_method, dust_redness in product(
#             filter_pairs, stacking_methods, dust_rednesses
#         ):
#             pref = ProductReference(
#                 kind=ProductKind.aperture_water_production,
#                 key=ContinuumSubtractionKey(
#                     epoch_id=eid.epoch_id,
#                     oh_filter=fp.oh_filter,
#                     dust_filter=fp.dust_filter,
#                     stacking_method=stacking_method,
#                     dust_redness_pct_per_hundred_nm=dust_redness,
#                 ),
#             )
#             terminal_products.append(pref)
#
#     return terminal_products


# def enumerate_radial_profile_water_production_products(
#     epochs: EpochIndex,
#     oh_filters: list[UvotFilter],
#     dust_filters: list[UvotFilter],
#     stacking_methods: list[StackingMethod],
#     dust_rednesses: list[DustReddeningPercent],
# ) -> list[ProductReference]:
#
#     terminal_products: list[ProductReference] = []
#     for eid in epochs:
#         filter_pairs = get_valid_water_production_filter_pairs(
#             eid=eid, oh_filters=oh_filters, dust_filters=dust_filters
#         )
#         for fp, stacking_method, dust_redness in product(
#             filter_pairs, stacking_methods, dust_rednesses
#         ):
#             pref = ProductReference(
#                 kind=ProductKind.radial_profile_water_production,
#                 key=ContinuumSubtractionKey(
#                     epoch_id=eid.epoch_id,
#                     oh_filter=fp.oh_filter,
#                     dust_filter=fp.dust_filter,
#                     stacking_method=stacking_method,
#                     dust_redness_pct_per_hundred_nm=dust_redness,
#                 ),
#             )
#             terminal_products.append(pref)
#
#     return terminal_products


# def enumerate_stacked_unbackgrounded_images(
#     epochs: EpochIndex,
# ) -> list[ProductReference]:
#     """
#     Returns a list of product references that can be stacked
#     """
#
#     stackable_list = []
#     for eid in epochs:
#         available_keys = [
#             EpochSubpipelineKey(epoch_id=eid.epoch_id, filter_type=f, stacking_method=s)
#             for f, s in product(
#                 UvotFilter.all_filters(), StackingMethod.all_stacking_methods()
#             )
#             if eid.exposure_times.get(f, 0) > 0.0
#         ]
#         stackable_products = [
#             ProductReference(kind=ProductKind.stacked_image_with_background, key=k)
#             for k in available_keys
#         ]
#         stackable_list.extend(stackable_products)
#
#     return stackable_list


def _is_data_ingestion_product(kind: ProductKind) -> bool:
    data_ingestion_products = [
        ProductKind.observation_log_raw,
        ProductKind.observation_log_with_epochs,
        ProductKind.observation_log_with_vetoes,
        ProductKind.orbit_data_earth,
        ProductKind.orbit_data_comet,
        ProductKind.epoch_index,
    ]
    return kind in data_ingestion_products


def _is_epoch_subpipeline_product(kind: ProductKind) -> bool:
    epoch_subpipeline_products = [
        ProductKind.stacked_image_with_background,
        ProductKind.stacked_image_exposure_map,
        ProductKind.background_determination,
        ProductKind.bg_subtracted_stacked_image,
        ProductKind.annular_aperture_photometry_analysis,
        ProductKind.radial_profile_from_cone,
        ProductKind.radial_profile_subtracted_image,
        ProductKind.afrho_from_aperture_photometry_analysis,
        ProductKind.afrho_from_radial_profile,
    ]
    return kind in epoch_subpipeline_products


def _is_continuum_subtraction_product(kind: ProductKind) -> bool:
    continuum_subtraction_products = [
        ProductKind.aperture_water_production,
        ProductKind.radial_profile_water_production,
    ]
    return kind in continuum_subtraction_products


def enumerate_subpipeline_product(
    kind: ProductKind,
    epochs: EpochIndex,
    # oh_filters: list[UvotFilter],
    # dust_filters: list[UvotFilter],
    filter_types: list[UvotFilter],
    stacking_methods: list[StackingMethod],
) -> list[ProductReference]:
    """
    Returns a list of product references of the given kind from all 'epochs' based on exposure times available for each filter
    """

    # all_filters = list(set(oh_filters + dust_filters))

    # remove duplicate filters
    all_filters = list(set(filter_types))

    prod_list = []
    for eid in epochs:
        available_keys = [
            EpochSubpipelineKey(epoch_id=eid.epoch_id, filter_type=f, stacking_method=s)
            for f, s in product(all_filters, stacking_methods)
            if eid.exposure_times.get(f, 0) > 0.0
        ]
        available_products = [
            ProductReference(kind=kind, key=k) for k in available_keys
        ]
        prod_list.extend(available_products)

    return prod_list


def enumerate_continuum_subtraction_products(
    kind: ProductKind,
    epochs: EpochIndex,
    oh_filters: list[UvotFilter],
    dust_filters: list[UvotFilter],
    stacking_methods: list[StackingMethod],
    dust_rednesses: list[DustReddeningPercent],
) -> list[ProductReference]:

    available_products: list[ProductReference] = []
    for eid in epochs:
        # look for valid combinations of oh and dust filters that have data this epoch
        filter_pairs = get_valid_water_production_filter_pairs(
            eid=eid, oh_filters=oh_filters, dust_filters=dust_filters
        )
        for fp, stacking_method, dust_redness in product(
            filter_pairs, stacking_methods, dust_rednesses
        ):
            pref = ProductReference(
                kind=kind,
                key=ContinuumSubtractionKey(
                    epoch_id=eid.epoch_id,
                    oh_filter=fp.oh_filter,
                    dust_filter=fp.dust_filter,
                    stacking_method=stacking_method,
                    dust_redness_pct_per_hundred_nm=dust_redness,
                ),
            )
            available_products.append(pref)

    return available_products


def enumerate_all_products_of(
    kind: ProductKind,
    epochs: EpochIndex,
    oh_filters: list[UvotFilter] | None = None,
    dust_filters: list[UvotFilter] | None = None,
    stacking_methods: list[StackingMethod] | None = None,
    dust_rednesses: list[DustReddeningPercent] | None = None,
) -> list[ProductReference]:
    """
    Take these parameters and build a list of ProductReferences that the EpochIndex says should exist,
    based on what filters have non-zero exposure times

    If you're asking for water production products, then only water based on combinations of oh_filters and dust_filters
    will be returned, not all possible combinations

    For things like radial profiles that come from one filter only, then we look for oh_filters+dust_filters and return
    them in the list without any distinction between oh/dust because that doesn't matter until we put them together for
    continuum subtraction
    """

    if _is_data_ingestion_product(kind=kind):
        return [ProductReference(kind=kind)]
    elif _is_epoch_subpipeline_product(kind=kind):
        if oh_filters is None or dust_filters is None or stacking_methods is None:
            print("Missing arguments for enumerate_all_products_of!")
            print(f"{oh_filters=}\t{dust_filters=}\t{stacking_methods=}")
            return []
        else:
            return enumerate_subpipeline_product(
                kind=kind,
                epochs=epochs,
                # oh_filters=oh_filters,
                # dust_filters=dust_filters,
                filter_types=oh_filters + dust_filters,
                stacking_methods=stacking_methods,
            )
    elif _is_continuum_subtraction_product(kind=kind):
        if (
            oh_filters is None
            or dust_filters is None
            or stacking_methods is None
            or dust_rednesses is None
        ):
            print("Missing arguments for enumerate_all_products_of!")
            print(
                f"{oh_filters=}\t{dust_filters=}\t{stacking_methods=}\t{dust_rednesses}"
            )
            return []
        else:
            return enumerate_continuum_subtraction_products(
                kind=kind,
                epochs=epochs,
                oh_filters=oh_filters,
                dust_filters=dust_filters,
                stacking_methods=stacking_methods,
                dust_rednesses=dust_rednesses,
            )
    else:
        # should be inaccessible
        print(f"Unhandled product kind in enumerate_all_products_of: {kind}")
        exit(1)
