from functools import partial

from joblib import Parallel, delayed
import astropy.units as u
from tqdm import tqdm

from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
)
from swift_comet_pipeline.modeling.vectorial.vectorial_model_cache import (
    vectorial_model_cache_init,
)
from swift_comet_pipeline.pipeline.product_enumeration import enumerate_all_products_of
from swift_comet_pipeline.pipeline.product_system.dependency_dag import (
    ProductBuildStatus,
    get_pipeline_status_for_product,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ContinuumSubtractionKey,
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.ui.tui.tui_common import (
    build_product_reference_loop,
    show_pipeline_status_for_product,
)


def build_all_data_ingestion_products(scp: Products) -> None:

    print("Checking data ingestion ...")
    epoch_index_ref = ProductReference(kind=ProductKind.epoch_index)
    build_product_reference_loop(scp=scp, ref=epoch_index_ref)
    show_pipeline_status_for_product(
        scp=scp, ref=epoch_index_ref, silent_if_complete=False
    )


def stack_all_images(scp: Products, epoch_index: EpochIndex) -> None:

    print(f"Stacking all images...")
    stackable_image_p_refs = enumerate_all_products_of(
        kind=ProductKind.stacked_image_with_background,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for stackable in stackable_image_p_refs:
        assert isinstance(stackable.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=stackable)
        build_product_reference_loop(scp=scp, ref=stackable)


def background_all_images(scp: Products, epoch_index: EpochIndex) -> None:

    print(f"Determining background of all images...")
    bg_refs = enumerate_all_products_of(
        kind=ProductKind.background_determination,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for bg in bg_refs:
        assert isinstance(bg.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=bg)
        build_product_reference_loop(scp=scp, ref=bg)


def background_subtract_all_images(scp: Products, epoch_index: EpochIndex) -> None:

    print("Background subtracting all images...")
    bg_sub_refs = enumerate_all_products_of(
        kind=ProductKind.bg_subtracted_stacked_image,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for bg_sub in bg_sub_refs:
        assert isinstance(bg_sub.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=bg_sub)
        build_product_reference_loop(scp=scp, ref=bg_sub)


def all_aperture_analysis(scp: Products, epoch_index: EpochIndex) -> None:
    print("Performing all aperture analysis...")
    aa_refs = enumerate_all_products_of(
        kind=ProductKind.annular_aperture_photometry_analysis,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for aa_ref in aa_refs:
        assert isinstance(aa_ref.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=aa_ref)
        build_product_reference_loop(scp=scp, ref=aa_ref)


def all_radial_profile_extraction(scp: Products, epoch_index: EpochIndex) -> None:
    print("Performing all radial profile extraction...")
    rp_refs = enumerate_all_products_of(
        kind=ProductKind.radial_profile_from_cone,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for rp_ref in rp_refs:
        assert isinstance(rp_ref.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=rp_ref)
        build_product_reference_loop(scp=scp, ref=rp_ref)


def all_radial_profile_subtracted_images(
    scp: Products, epoch_index: EpochIndex
) -> None:
    print("Generating all radial profile subtraction images...")
    rp_refs = enumerate_all_products_of(
        kind=ProductKind.radial_profile_subtracted_image,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for rp_ref in rp_refs:
        assert isinstance(rp_ref.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=rp_ref)
        build_product_reference_loop(scp=scp, ref=rp_ref)


def all_afrho_from_apertures(scp: Products, epoch_index: EpochIndex) -> None:
    print("Calculating afrho from aperture photometry...")
    afrho_refs = enumerate_all_products_of(
        kind=ProductKind.afrho_from_aperture_photometry_analysis,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for afrho_ref in afrho_refs:
        assert isinstance(afrho_ref.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=afrho_ref)
        build_product_reference_loop(scp=scp, ref=afrho_ref)


def all_afrho_from_radial_profiles(scp: Products, epoch_index: EpochIndex) -> None:
    print("Calculating afrho from radial profiles...")
    afrho_refs = enumerate_all_products_of(
        kind=ProductKind.afrho_from_radial_profile,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=scp.cfg.stacking_methods,
    )
    for afrho_ref in afrho_refs:
        assert isinstance(afrho_ref.key, EpochSubpipelineKey)
        show_pipeline_status_for_product(scp=scp, ref=afrho_ref)
        build_product_reference_loop(scp=scp, ref=afrho_ref)


def parallel_loop_builder(scp: Products, ref: ProductReference, force: bool) -> None:
    vectorial_model_settings_init(comet_project_config=scp.cfg)
    build_product_reference_loop(scp=scp, ref=ref, force=force)


# TODO: add forcing to the other functions
def all_aperture_water_analysis(
    scp: Products, epoch_index: EpochIndex, force: bool = False
) -> None:

    print(f"Performing all aperture water analysis for:")
    for eid in epoch_index:
        print(
            f"{eid.epoch_id} --> {eid.observation_time} | {eid.rh_au} AU | T-Tp: {eid.time_from_perihelion.to_value(u.day)} days"  # type: ignore
        )

    # aperture analysis water production
    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.aperture_water_production,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=[StackingMethod.summation, StackingMethod.median],
        dust_rednesses=scp.dust_rednesses,
    )

    incomplete_awp_prefs = list(
        filter(
            lambda p: get_pipeline_status_for_product(scp=scp, ref=p)
            != ProductBuildStatus.complete,
            awp_prefs,
        )
    )

    if force:
        product_build_list = awp_prefs
    else:
        product_build_list = incomplete_awp_prefs

    # print(f"We have {len(product_build_list)} items to build.  Force is set to {force}")
    # for awp in product_build_list:
    #     assert isinstance(awp.key, ContinuumSubtractionKey)
    #     show_pipeline_status_for_product(scp=scp, ref=awp)
    #     build_product_reference_loop(scp=scp, ref=awp, force=force)
    #     # test_aperture_water_analysis_loading(scp=scp, ref=awp)

    # builder_func = partial(build_product_reference_loop, scp=scp, force=force)
    builder_func = partial(parallel_loop_builder, scp=scp, force=force)
    Parallel(n_jobs=-1, backend="loky")(
        delayed(builder_func)(ref=x) for x in product_build_list
    )


def all_radial_profile_water_analysis(scp: Products, epoch_index: EpochIndex) -> None:

    # vectorial model/radial profile water production
    rwp_prefs = enumerate_all_products_of(
        kind=ProductKind.radial_profile_water_production,
        epochs=epoch_index,
        oh_filters=scp.cfg.oh_filters,
        dust_filters=scp.cfg.dust_filters,
        stacking_methods=[StackingMethod.summation],
        dust_rednesses=scp.dust_rednesses,
    )
    incomplete_rwp_prefs = list(
        filter(
            lambda p: get_pipeline_status_for_product(scp=scp, ref=p)
            != ProductBuildStatus.complete,
            rwp_prefs,
        )
    )
    for rwp in tqdm(incomplete_rwp_prefs, total=len(incomplete_rwp_prefs)):
        assert isinstance(rwp.key, ContinuumSubtractionKey)
        build_product_reference_loop(scp=scp, ref=rwp)
