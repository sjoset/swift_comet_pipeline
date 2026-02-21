from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from swift_comet_pipeline.image_manipulation.utility.plot_image_multi import (
    plot_images_multi,
)
from swift_comet_pipeline.pipeline.product_enumeration import enumerate_all_products_of
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import (
    BayesianPriorBlueSpotLightcurveKey,
    BayesianPriorLightcurveKey,
    BlueSpotLightcurveKey,
    ContinuumSubtractionKey,
    EpochSubpipelineKey,
    LightcurveKey,
)
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.compound.lightcurve import (
    bayesian_prior_blue_spot_lightcurve_to_dataframe,
    blue_spot_lightcurve_to_dataframe,
    lightcurve_to_dataframe,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_aperture_photometry import (
    dataframe_from_afrho_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_profile import (
    dataframe_from_afrho_from_radial_profile,
)
from swift_comet_pipeline.scp_types.primitive.annular_aperture_photometry_analysis import (
    dataframe_from_annular_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive.aperture_water_production_analysis import (
    dataframe_from_aperture_water_production_analysis,
)
from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter
from swift_comet_pipeline.ui.tui.tui_common import build_product_reference_loop


def test_afrho_profile_plotting(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    dust_redness: DustReddeningPercent,
) -> None:

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.afrho_from_radial_profile,
        epochs=[eid],
        oh_filters=[oh_filter],
        dust_filters=[dust_filter],
        stacking_methods=[StackingMethod.summation],
        dust_rednesses=[dust_redness],
    )
    build_product_reference_loop(scp=scp, ref=awp_prefs[0])

    pkey = awp_prefs[0].key
    assert isinstance(pkey, EpochSubpipelineKey)

    afrp = scp.load_afrho_from_profile(key=pkey)
    assert afrp is not None
    afrp_df = dataframe_from_afrho_from_radial_profile(afrp=afrp)

    plt.plot(afrp_df.r_km, np.log10(afrp_df.cumulative_afrho_zero_cm))
    # plt.plot(afrp_df.r_km, afrp_df.cumulative_afrho_cm)
    plt.show()


def test_afrho_aperture_plotting(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    dust_redness: DustReddeningPercent,
) -> None:

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.afrho_from_aperture_photometry_analysis,
        epochs=[eid],
        oh_filters=[oh_filter],
        dust_filters=[dust_filter],
        stacking_methods=[StackingMethod.summation],
        dust_rednesses=[dust_redness],
    )
    build_product_reference_loop(scp=scp, ref=awp_prefs[0])

    pkey = awp_prefs[0].key
    assert isinstance(pkey, EpochSubpipelineKey)

    afapa = scp.load_afrho_from_aperture_photometry(key=pkey)
    afapa_df = dataframe_from_afrho_aperture_photometry_analysis(afapa=afapa)

    plt.plot(afapa_df.aperture_r_km, np.log10(afapa_df.cumulative_afrho_zero_median_cm))
    plt.plot(afapa_df.aperture_r_km, np.log10(afapa_df.cumulative_afrho_zero_sum_cm))

    plt.show()


def test_radial_water_production_plotting(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    dust_redness: DustReddeningPercent,
) -> None:

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.radial_profile_water_production,
        epochs=[eid],
        oh_filters=[oh_filter],
        dust_filters=[dust_filter],
        stacking_methods=[StackingMethod.summation],
        dust_rednesses=[dust_redness],
    )
    build_product_reference_loop(scp=scp, ref=awp_prefs[0])

    pkey = awp_prefs[0].key
    assert isinstance(pkey, ContinuumSubtractionKey)

    rpwpa = scp.load_radial_profile_water_production_analysis(key=pkey)

    left_edge = scp.cfg.near_far_split_radius_km
    right_edge = np.max(rpwpa.oh_column_density.rs_km)
    plt.axvspan(
        left_edge, right_edge, color="orange", alpha=0.07, label="fitting region"
    )

    plt.plot(
        rpwpa.oh_column_density.rs_km, rpwpa.oh_column_density.cd_cm2, label="data"
    )
    plt.plot(
        rpwpa.near_fit.vectorial_column_density.rs_km,
        rpwpa.near_fit.vectorial_column_density.cd_cm2,
        label="near",
        alpha=0.7,
    )
    plt.plot(
        rpwpa.far_fit.vectorial_column_density.rs_km,
        rpwpa.far_fit.vectorial_column_density.cd_cm2,
        label="far",
        alpha=0.7,
    )
    plt.plot(
        rpwpa.full_fit.vectorial_column_density.rs_km,
        rpwpa.full_fit.vectorial_column_density.cd_cm2,
        label="full",
        alpha=0.7,
    )
    plt.legend()

    # water_df = dataframe_from_aperture_water_production_analysis(awpa=awpa)

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.aperture_matched_q_h2o_sum
    #         + i * water_df.aperture_matched_q_h2o_sum_err,
    #         water_df.aperture_matched_q_h2o_sum
    #         - i * water_df.aperture_matched_q_h2o_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # awp_prefs = enumerate_aperture_water_production_products(
    #     epochs=[eid],
    #     oh_filters=[oh_filter],
    #     dust_filters=[dust_filter],
    #     stacking_methods=[StackingMethod.summation],
    #     dust_rednesses=[dust_redness],
    # )

    plt.yscale("log")
    plt.xscale("log")
    plt.title(
        f"Q: {rpwpa.far_fit.best_fit_q_per_s:3.3e}  OH filter: {oh_filter.value}  Dust filter: {dust_filter.value}"
    )

    # plt.xlim(0, 500000)
    # plt.ylim(
    #     1e26,
    # )
    plt.show()


def test_aperture_water_analysis_plotting(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    dust_redness: DustReddeningPercent,
):

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.aperture_water_production,
        epochs=[eid],
        oh_filters=[oh_filter],
        dust_filters=[dust_filter],
        stacking_methods=[stacking_method],
        dust_rednesses=[dust_redness],
    )
    if not scp.exists(awp_prefs[0]):
        build_product_reference_loop(scp=scp, ref=awp_prefs[0])

    pkey = awp_prefs[0].key
    assert isinstance(pkey, ContinuumSubtractionKey)

    awpa = scp.load_aperture_water_production_analysis(key=pkey)
    water_df = dataframe_from_aperture_water_production_analysis(awpa=awpa)

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.aperture_matched_q_h2o_sum
    #         + i * water_df.aperture_matched_q_h2o_sum_err,
    #         water_df.aperture_matched_q_h2o_sum
    #         - i * water_df.aperture_matched_q_h2o_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.equivalent_q_h2o_sum + i * water_df.equivalent_q_h2o_sum_err,
    #         water_df.equivalent_q_h2o_sum - i * water_df.equivalent_q_h2o_sum_err,
    #         alpha=0.2,
    #         color="#dbb89c",
    #         # color="#c74a77",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.equivalent_q_h2o_median + i * water_df.equivalent_q_h2o_median_err,
    #         water_df.equivalent_q_h2o_median - i * water_df.equivalent_q_h2o_median_err,
    #         alpha=0.2,
    #         color="#dbb89c",
    #         # color="#c74a77",
    #     )

    for i in range(4):
        plt.fill_between(
            water_df.aperture_r_km,
            water_df.aperture_matched_q_h2o_sum
            + i * water_df.aperture_matched_q_h2o_sum_err,
            water_df.aperture_matched_q_h2o_sum
            - i * water_df.aperture_matched_q_h2o_sum_err,
            alpha=0.2,
            color="#688894",
        )

    for i in range(4):
        plt.fill_between(
            water_df.aperture_r_km,
            water_df.aperture_matched_q_h2o_median
            + i * water_df.aperture_matched_q_h2o_median_err,
            water_df.aperture_matched_q_h2o_median
            - i * water_df.aperture_matched_q_h2o_median_err,
            alpha=0.2,
            color="#afac7c",
        )

    for i in range(4):
        plt.fill_between(
            water_df.aperture_r_km,
            water_df.equivalent_q_h2o_sum + i * water_df.equivalent_q_h2o_sum_err,
            water_df.equivalent_q_h2o_sum - i * water_df.equivalent_q_h2o_sum_err,
            alpha=0.2,
            color="#301e2a",
        )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.equivalent_q_h2o_median + i * water_df.equivalent_q_h2o_median_err,
    #         water_df.equivalent_q_h2o_median - i * water_df.equivalent_q_h2o_median_err,
    #         alpha=0.2,
    #         color="#afac7c",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.cumulative_sum_magnitude
    #         + i * water_df.cumulative_sum_magnitude_variance,
    #         water_df.cumulative_sum_magnitude
    #         - i * water_df.cumulative_sum_magnitude_variance,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         dust_aapa_df.aperture_r_km,
    #         dust_aapa_df.cumulative_sum_magnitude
    #         + i * dust_aapa_df.cumulative_sum_magnitude_variance,
    #         dust_aapa_df.cumulative_sum_magnitude
    #         - i * dust_aapa_df.cumulative_sum_magnitude_variance,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.cumulative_median_magnitude
    #         + i * water_df.cumulative_median_magnitude_variance,
    #         water_df.cumulative_median_magnitude
    #         - i * water_df.cumulative_median_magnitude_variance,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         dust_aapa_df.aperture_r_km,
    #         dust_aapa_df.cumulative_median_magnitude
    #         + i * dust_aapa_df.cumulative_median_magnitude_variance,
    #         dust_aapa_df.cumulative_median_magnitude
    #         - i * dust_aapa_df.cumulative_median_magnitude_variance,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.num_oh_sum + i * water_df.num_oh_sum_err,
    #         water_df.num_oh_sum - i * water_df.num_oh_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_flux_sum + i * water_df.oh_flux_sum_err,
    #         water_df.oh_flux_sum - i * water_df.oh_flux_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_counts_sum + i * water_df.oh_counts_sum_err,
    #         water_df.oh_counts_sum - i * water_df.oh_counts_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_counts_median + i * water_df.oh_counts_median_err,
    #         water_df.oh_counts_median - i * water_df.oh_counts_median_err,
    #         alpha=0.2,
    #         color="#afac7c",
    #     )

    plt.yscale("log")
    # plt.xscale("log")

    # plt.xlim(0, 500000)
    # plt.ylim(
    #     1e26,
    # )
    plt.title(f"OH filter: {str(oh_filter)}  Dust filter: {str(dust_filter)}")
    # plt.show()
    return plt.gcf()


def test_radial_profile_smoothing(
    scp: Products,
    eid: EpochIndexEntry,
    filter_type: UvotFilter,
    stacking_method: StackingMethod = StackingMethod.summation,
) -> None:

    pkey = EpochSubpipelineKey(
        epoch_id=eid.epoch_id, filter_type=filter_type, stacking_method=stacking_method
    )
    crpfc = scp.load_extracted_radial_profile(key=pkey)

    xs = crpfc.profile_axis_rs
    ys = crpfc.pixel_values

    smoothed_ys = savgol_filter(ys, window_length=10, polyorder=2)

    plt.plot(xs, smoothed_ys, label="smoothed")  # type: ignore
    plt.plot(crpfc.profile_axis_rs, crpfc.pixel_values, label="raw")
    # plt.plot(xs, crpfc.pixel_values / smoothed_ys, label="raw to smooth")

    plt.legend()
    plt.show()


def test_aperture_analysis_loading(
    scp: Products, eid: EpochIndexEntry, filter_type: UvotFilter
) -> None:
    """
    Load and plot count rates etc. as a function of radius
    """

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.annular_aperture_photometry_analysis,
        epochs=[eid],
        oh_filters=[filter_type],
        dust_filters=[filter_type],
        stacking_methods=[StackingMethod.summation],
    )
    build_product_reference_loop(scp=scp, ref=awp_prefs[0])

    pkey = awp_prefs[0].key
    assert isinstance(pkey, EpochSubpipelineKey)

    aapa_load_res = scp.load_annular_aperture_analysis(key=pkey)
    assert aapa_load_res is not None
    aapa, metadata = aapa_load_res
    print(f"Aperture analysis metadata: {metadata}")
    aapa_df = dataframe_from_annular_aperture_photometry_analysis(aapa=aapa)

    smoothed_total = savgol_filter(
        aapa_df.cumulative_sum, window_length=20, polyorder=2
    )
    smoothed_median = savgol_filter(
        aapa_df.cumulative_area_scaled_median, window_length=20, polyorder=2
    )

    plt.plot(
        aapa_df.aperture_r_km, smoothed_total, label="smooth total", color="#688894"  # type: ignore
    )
    plt.plot(
        aapa_df.aperture_r_km, smoothed_median, label="smooth median", color="#afac7c"  # type: ignore
    )

    for i in range(4):
        plt.fill_between(
            aapa_df.aperture_r_km,
            aapa_df.cumulative_sum - i * aapa_df.cumulative_sum_err,
            aapa_df.cumulative_sum + i * aapa_df.cumulative_sum_err,
            alpha=0.2,
            color="#688894",
        )
        plt.fill_between(
            aapa_df.aperture_r_km,
            aapa_df.cumulative_area_scaled_median
            - i * aapa_df.cumulative_area_scaled_median_err,
            aapa_df.cumulative_area_scaled_median
            + i * aapa_df.cumulative_area_scaled_median_err,
            alpha=0.2,
            color="#afac7c",
        )

    plt.title(f"Filter: {filter_type.value}")
    plt.legend()
    plt.xlim(0, 1000000)
    plt.ylim(
        0,
    )
    plt.show()


# TODO: make this take arguments
def test_background_result_loading(scp: Products) -> None:
    target_ref = ProductReference(
        kind=ProductKind.background_determination,
        key=EpochSubpipelineKey(
            epoch_id="004_2015_Jun_19",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    pkey = target_ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    bgr = scp.load_background_result(key=pkey)
    print(f"Background for {target_ref}: {bgr}")


# TODO: make this take arguments
def test_fits_loading(scp: Products) -> None:

    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            # epoch_id="000_2014_Aug_14",
            epoch_id="004_2015_Jun_19",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    fits_sum = scp.load_fits_image(ref=target_ref)
    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            # epoch_id="000_2014_Aug_14",
            epoch_id="004_2015_Jun_19",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.median,
        ),
    )
    fits_median = scp.load_fits_image(ref=target_ref)

    if fits_sum is None or fits_median is None:
        print(f"fits image of {target_ref} failed to load!")
        return

    _ = plot_images_multi(images=[fits_sum.data, fits_median.data], comet_centers=None)
    plt.show()


def test_obs_log_metadata(scp: Products) -> None:

    obs_log = scp.load_raw_log()
    if obs_log is None:
        print("Failed to load observation log")
        return

    print(f"Raw log metadata: {obs_log.attrs}")

    obs_log = scp.load_obs_log()
    if obs_log is None:
        print("Failed to load observation log")
        return

    print(f"Log metadata: {obs_log.attrs}")


def test_epoch_index_loading(scp: Products) -> None:
    epoch_index = scp.load_epoch_index()
    if not epoch_index:
        print("no epoch index")
        return

    for epoch in epoch_index:
        print("----")
        print(
            epoch.epoch_id,
            "\t",
            epoch.epoch_length,
            "\t",
            epoch.time_from_perihelion,
            "\t",
            epoch.observation_time,
        )


def test_lightcurve_loading(scp: Products):
    lc_ref = ProductReference(
        kind=ProductKind.water_production_lightcurve,
        key=LightcurveKey(
            oh_filter=UvotFilter.uw1,
            dust_filter=UvotFilter.uvv,
            stacking_method=StackingMethod.summation,
        ),
    )

    assert isinstance(lc_ref.key, LightcurveKey)
    lc = scp.load_water_production_lightcurve(key=lc_ref.key)
    print(f"lightcurve:")
    print(lc[0], lc[1])
    df = lightcurve_to_dataframe(lc=lc)
    print("Converted regular lightcurve to dataframe without errors!")
    print(f"Columns: {df.columns}")


def test_bayesian_lightcurve_loading(scp: Products) -> None:
    blc_ref = ProductReference(
        kind=ProductKind.bayesian_water_production_lightcurve,
        key=BayesianPriorLightcurveKey(
            oh_filter=UvotFilter.uw1,
            dust_filter=UvotFilter.uvv,
            stacking_method=StackingMethod.summation,
            dust_redness_sigma_pct_per_hundred_nm=DustReddeningPercent(3.0),
        ),
    )

    assert isinstance(blc_ref.key, BayesianPriorLightcurveKey)
    blc = scp.load_bayesian_water_production_lightcurve(key=blc_ref.key)
    print(blc[0])
    bdf = lightcurve_to_dataframe(lc=blc)
    print(f"Dataframe columns: {bdf.columns}")


def test_blue_spot_lightcurve_loading(scp: Products) -> None:

    # TODO: fix these up to take arguments
    bs_ref = ProductReference(
        kind=ProductKind.blue_spot_lightcurve,
        key=BlueSpotLightcurveKey(
            oh_filter=UvotFilter.uw1,
            dust_filter=UvotFilter.uvv,
            stacking_method=StackingMethod.summation,
            blue_spot_extent_km=20000,
        ),
    )
    assert isinstance(bs_ref.key, BlueSpotLightcurveKey)

    bslc = scp.load_blue_spot_lightcurve(key=bs_ref.key)
    # print(bslc)
    bsdf = blue_spot_lightcurve_to_dataframe(lc=bslc)
    print(bsdf.columns)


def test_bayesian_blue_spot_lightcurve(scp: Products) -> None:

    for bse, sig in product(scp.blue_spot_extents_km, scp.cfg.bayesian_prior_sigmas):
        bpbs_ref = ProductReference(
            kind=ProductKind.bayesian_blue_spot_lightcurve,
            key=BayesianPriorBlueSpotLightcurveKey(
                oh_filter=UvotFilter.uw1,
                dust_filter=UvotFilter.uvv,
                stacking_method=StackingMethod.summation,
                blue_spot_extent_km=bse,
                dust_redness_sigma_pct_per_hundred_nm=sig,
            ),
        )
        assert isinstance(bpbs_ref.key, BayesianPriorBlueSpotLightcurveKey)

        bslc = scp.load_bayesian_blue_spot_lightcurve(key=bpbs_ref.key)
        # print(bslc)
        bsdf = bayesian_prior_blue_spot_lightcurve_to_dataframe(lc=bslc)
        print(bsdf.columns)
