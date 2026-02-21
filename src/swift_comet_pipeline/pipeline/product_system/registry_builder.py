from functools import partial
from itertools import product

from swift_comet_pipeline.pipeline.product_enumeration import (
    enumerate_continuum_subtraction_products,
)
from swift_comet_pipeline.pipeline.product_system.codecs import *
from swift_comet_pipeline.pipeline.product_system.product_key import *
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ProductRegistry,
    ProductSpecification,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# Default registry with central key→subdir policy
# -----------------------------------------------------------------------------
def data_ingestion_registry() -> ProductRegistry:
    reg = ProductRegistry()

    reg.subdir_resolver().register_template(GlobalKey, ".")
    reg.subdir_resolver().register_template(
        EpochSubpipelineKey,
        "{key.epoch_id}/{key.filter_type}/{key.stacking_method}",
    )
    reg.subdir_resolver().register_template(
        ContinuumSubtractionKey,
        "{key.epoch_id}/oh_{key.oh_filter}_dust_{key.dust_filter}/{key.stacking_method}/{key.dust_redness_pct_per_hundred_nm:06.2f}",
    )
    reg.subdir_resolver().register_template(
        LightcurveKey,
        subdir_template="lightcurves/oh_{key.oh_filter}_dust_{key.dust_filter}_{key.stacking_method}",
    )
    reg.subdir_resolver().register_template(
        BayesianPriorLightcurveKey,
        subdir_template="lightcurves/bayesian_oh_{key.oh_filter}_dust_{key.dust_filter}_{key.stacking_method}_prior_sigma_{key.dust_redness_sigma_pct_per_hundred_nm:06.2f}",
    )
    reg.subdir_resolver().register_template(
        BlueSpotLightcurveKey,
        subdir_template="lightcurves/blue_spot_oh_{key.oh_filter}_dust_{key.dust_filter}_{key.stacking_method}_{key.blue_spot_extent_km:06.2f}",
    )
    reg.subdir_resolver().register_template(
        BayesianPriorBlueSpotLightcurveKey,
        subdir_template="lightcurves/bayesian_blue_spot_oh_{key.oh_filter}_dust_{key.dust_filter}_{key.stacking_method}_prior_sigma_{key.dust_redness_sigma_pct_per_hundred_nm:06.2f}_spot_extent_{key.blue_spot_extent_km:06.2f}",
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()),
            filename_stem_template="observation_log_raw",
            codec=ObservationLogCodec(),
            deps=None,
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
            ),
            filename_stem_template="observation_log_with_epochs",
            codec=ObservationLogCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey())
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(
                kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
            ),
            filename_stem_template="observation_log_with_vetoes",
            codec=ObservationLogCodec(),
            deps=lambda _: [
                ProductReference(
                    kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
                )
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey()),
            filename_stem_template="orbital_data_earth",
            codec=PandasDataframeToCSVCodec(),
            deps=lambda _: [
                ProductReference(
                    kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
                )
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey()),
            filename_stem_template="orbital_data_comet",
            codec=PandasDataframeToCSVCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey())
            ],
        )
    )

    reg.register(
        ProductSpecification(
            ref=ProductReference(kind=ProductKind.epoch_index, key=GlobalKey()),
            filename_stem_template="epoch_index",
            codec=JSONCodec(),
            deps=lambda _: [
                ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey())
            ],
        )
    )

    return reg


def add_epoch_subpipelines_to_registry(
    reg: ProductRegistry, epoch_index: EpochIndex
) -> None:
    """
    After the epoch index has been built, we look at which filters have exposure times and add subpipelines for those images
    """

    for epoch in epoch_index:

        epoch_id = epoch.epoch_id
        epoch_subpipe_key_func = partial(EpochSubpipelineKey, epoch_id=epoch_id)

        for filter_type, stacking_method in product(
            epoch.exposure_times.keys(), StackingMethod.all_stacking_methods()
        ):
            epoch_subpipe_key = epoch_subpipe_key_func(
                filter_type=filter_type, stacking_method=stacking_method
            )

            stacked_image_with_bg_ref = ProductReference(
                kind=ProductKind.stacked_image_with_background, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=stacked_image_with_bg_ref,
                    filename_stem_template="stack_with_background",
                    codec=InternalFITSImageCodec(),
                    deps=lambda _: [
                        ProductReference(kind=ProductKind.epoch_index, key=GlobalKey())
                    ],
                )
            )

            exp_map_ref = ProductReference(
                kind=ProductKind.stacked_image_exposure_map, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=exp_map_ref,
                    filename_stem_template="exposure_map",
                    codec=InternalFITSImageCodec(),
                    deps=lambda _: [ProductReference(kind=ProductKind.epoch_index)],
                )
            )

            bg_ref = ProductReference(
                kind=ProductKind.background_determination, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=bg_ref,
                    filename_stem_template="background_determination",
                    codec=JSONCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.stacked_image_exposure_map, key=p_ref.key
                        )
                    ],
                )
            )

            bg_sub_ref = ProductReference(
                kind=ProductKind.bg_subtracted_stacked_image, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=bg_sub_ref,
                    filename_stem_template="bg_subtracted_stack",
                    codec=InternalFITSImageCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.background_determination, key=p_ref.key
                        )
                    ],
                )
            )

            ap_phot_ref = ProductReference(
                ProductKind.annular_aperture_photometry_analysis, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=ap_phot_ref,
                    filename_stem_template="aperture_photometry_analysis",
                    codec=PandasDataframeToECSVCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.bg_subtracted_stacked_image, key=p_ref.key
                        )
                    ],
                )
            )

            afrho_ap_ref = ProductReference(
                ProductKind.afrho_from_aperture_photometry_analysis,
                key=epoch_subpipe_key,
            )
            reg.register(
                spec=ProductSpecification(
                    ref=afrho_ap_ref,
                    filename_stem_template="afrho_from_aperture_photometry",
                    codec=PandasDataframeToECSVCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.annular_aperture_photometry_analysis,
                            key=p_ref.key,
                        )
                    ],
                )
            )

            cone_prof_ref = ProductReference(
                ProductKind.radial_profile_from_cone, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=cone_prof_ref,
                    filename_stem_template="radial_profile_from_cone",
                    codec=JSONCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.bg_subtracted_stacked_image, key=p_ref.key
                        )
                    ],
                )
            )

            prof_sub_ref = ProductReference(
                ProductKind.radial_profile_subtracted_image, key=epoch_subpipe_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=prof_sub_ref,
                    filename_stem_template="radial_profile_subtracted",
                    codec=InternalFITSImageCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.radial_profile_from_cone, key=p_ref.key
                        )
                    ],
                )
            )

            afrho_prof_ref = ProductReference(
                ProductKind.afrho_from_radial_profile,
                key=epoch_subpipe_key,
            )
            reg.register(
                spec=ProductSpecification(
                    ref=afrho_prof_ref,
                    filename_stem_template="afrho_from_radial_profile",
                    codec=PandasDataframeToECSVCodec(),
                    deps=lambda p_ref: [
                        ProductReference(
                            kind=ProductKind.radial_profile_from_cone,
                            key=p_ref.key,
                        )
                    ],
                )
            )

    return


# TODO: this should respect stacking methods passed in through the config file - but it does no harm
# to add them to the registry - when we ask for things to be built, we only look at the requested stacking methods anyway.
def add_continuum_subtraction_products_to_registry(
    reg: ProductRegistry,
    epoch_index: EpochIndex,
    oh_filters: list[UvotFilter],
    dust_filters: list[UvotFilter],
    dust_rednesses: list[DustReddeningPercent],
) -> None:

    oh_and_dust_filter_combinations = list(product(oh_filters, dust_filters))

    for eid in epoch_index:
        wkey_func = partial(ContinuumSubtractionKey, epoch_id=eid.epoch_id)

        for oh_filter, dust_filter in oh_and_dust_filter_combinations:
            # check if each filter is in this epoch
            if (
                not oh_filter in eid.exposure_times
                or not dust_filter in eid.exposure_times
            ):
                # TODO: debug log that we skipped this and why
                continue

            # print(f"Found {oh_filter} and {dust_filter} in {eid.epoch_id}")
            for stacking_method, dust_redness in product(
                StackingMethod.all_stacking_methods(), dust_rednesses
            ):
                wkey = wkey_func(
                    oh_filter=oh_filter,
                    dust_filter=dust_filter,
                    stacking_method=stacking_method,
                    dust_redness_pct_per_hundred_nm=dust_redness,
                )

                ap_parent_kind = ProductKind.annular_aperture_photometry_analysis
                ap_deps = lambda p_ref: [
                    ProductReference(
                        kind=ap_parent_kind,
                        key=EpochSubpipelineKey(
                            epoch_id=p_ref.key.epoch_id,
                            filter_type=f,
                            stacking_method=p_ref.key.stacking_method,
                        ),
                    )
                    for f in [p_ref.key.oh_filter, p_ref.key.dust_filter]
                ]

                ap_wat_ref = ProductReference(
                    kind=ProductKind.aperture_water_production, key=wkey
                )
                reg.register(
                    spec=ProductSpecification(
                        ref=ap_wat_ref,
                        filename_stem_template="aperture_water_production",
                        codec=PandasDataframeToECSVCodec(),
                        deps=ap_deps,
                    )
                )

                rad_parent_kind = ProductKind.radial_profile_from_cone
                rad_deps = lambda p_ref: [
                    ProductReference(
                        kind=rad_parent_kind,
                        key=EpochSubpipelineKey(
                            epoch_id=p_ref.key.epoch_id,
                            filter_type=f,
                            stacking_method=p_ref.key.stacking_method,
                        ),
                    )
                    for f in [p_ref.key.oh_filter, p_ref.key.dust_filter]
                ]

                rad_wat_ref = ProductReference(
                    kind=ProductKind.radial_profile_water_production, key=wkey
                )
                reg.register(
                    spec=ProductSpecification(
                        ref=rad_wat_ref,
                        filename_stem_template="radial_profile_water_production",
                        codec=JSONCodec(),
                        deps=rad_deps,
                    )
                )


def add_lightcurve_products_to_registry(
    reg: ProductRegistry,
    epoch_index: EpochIndex,
    oh_filters: list[UvotFilter],
    dust_filters: list[UvotFilter],
    dust_rednesses: list[DustReddeningPercent],
    bayesian_prior_sigmas: list[DustReddeningPercent],
    blue_spot_extents_km: list[float],
) -> None:

    oh_dust_and_stacking_combinations = list(
        product(oh_filters, dust_filters, StackingMethod.all_stacking_methods())
    )
    all_epochs = epoch_index
    all_rednesses = dust_rednesses

    for oh_filter, dust_filter, stacking_method in oh_dust_and_stacking_combinations:
        lc_key = LightcurveKey(
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
        )

        # we want all vectorial and aperture products to be complete before building lightcurve
        vec_dep_list = enumerate_continuum_subtraction_products(
            kind=ProductKind.radial_profile_water_production,
            epochs=all_epochs,
            oh_filters=[oh_filter],
            dust_filters=[dust_filter],
            stacking_methods=[stacking_method],
            dust_rednesses=all_rednesses,
        )
        ap_dep_list = enumerate_continuum_subtraction_products(
            kind=ProductKind.aperture_water_production,
            epochs=all_epochs,
            oh_filters=[oh_filter],
            dust_filters=[dust_filter],
            stacking_methods=[stacking_method],
            dust_rednesses=all_rednesses,
        )

        lc_deps = lambda _: [x for x in vec_dep_list + ap_dep_list]

        # regular lightcurves as a function of redness, no bayesian color analysis
        lc_ref = ProductReference(
            kind=ProductKind.water_production_lightcurve, key=lc_key
        )
        reg.register(
            spec=ProductSpecification(
                ref=lc_ref,
                filename_stem_template="water_production_lightcurve",
                codec=PandasDataframeToCSVCodec(),
                deps=lc_deps,
            )
        )

        # regular blue-spot lightcurves
        for bs_extent in blue_spot_extents_km:
            bs_key = BlueSpotLightcurveKey(
                oh_filter=oh_filter,
                dust_filter=dust_filter,
                stacking_method=stacking_method,
                blue_spot_extent_km=bs_extent,
            )
            bs_ref = ProductReference(kind=ProductKind.blue_spot_lightcurve, key=bs_key)
            reg.register(
                spec=ProductSpecification(
                    ref=bs_ref,
                    filename_stem_template="blue_spot_lightcurve",
                    codec=PandasDataframeToCSVCodec(),
                    deps=lc_deps,
                )
            )

        # lightcurves as a function of the mean redness of the prior, which means the prior's sigma is free to vary.
        # for each bayesian prior sigma, we want another lightcurve product, and it will depend on the same
        # vectorial/aperture water production analysis to be finished as the normal lightcurve
        for psig in bayesian_prior_sigmas:
            blc_key = BayesianPriorLightcurveKey(
                oh_filter=oh_filter,
                dust_filter=dust_filter,
                stacking_method=stacking_method,
                dust_redness_sigma_pct_per_hundred_nm=psig,
            )
            blc_ref = ProductReference(
                kind=ProductKind.bayesian_water_production_lightcurve, key=blc_key
            )
            reg.register(
                spec=ProductSpecification(
                    ref=blc_ref,
                    filename_stem_template="bayesian_water_production_lightcurve",
                    codec=PandasDataframeToCSVCodec(),
                    deps=lc_deps,
                )
            )

            for bs_extent in blue_spot_extents_km:
                bpbs_key = BayesianPriorBlueSpotLightcurveKey(
                    oh_filter=oh_filter,
                    dust_filter=dust_filter,
                    stacking_method=stacking_method,
                    blue_spot_extent_km=bs_extent,
                    dust_redness_sigma_pct_per_hundred_nm=psig,
                )
                bpbs_ref = ProductReference(
                    kind=ProductKind.bayesian_blue_spot_lightcurve, key=bpbs_key
                )
                reg.register(
                    spec=ProductSpecification(
                        ref=bpbs_ref,
                        filename_stem_template="bayesian_blue_spot_lightcurve",
                        codec=PandasDataframeToCSVCodec(),
                        deps=lc_deps,
                    )
                )
