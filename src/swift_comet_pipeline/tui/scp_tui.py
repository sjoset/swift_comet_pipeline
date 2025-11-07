#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log
from argparse import ArgumentParser

import pandas as pd
from scipy.signal import savgol_filter
from astropy.io.fits.card import VerifyWarning
from astropy.wcs.wcs import FITSFixedWarning
from pandas.errors import SettingWithCopyWarning
from rich.console import Console
import matplotlib.pyplot as plt
from tqdm import tqdm

from swift_comet_pipeline.builders.build_dispatcher import do_build
from swift_comet_pipeline.image_manipulation.utility.plot_image_multi import (
    plot_images_multi,
)
from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
)
from swift_comet_pipeline.photometry.dust.reddening_translate import (
    demo_reddening_recalculation,
)
from swift_comet_pipeline.pipeline.product_enumeration import enumerate_all_products_of
from swift_comet_pipeline.pipeline.project_configuration.read_comet_project_config import (
    read_comet_project_config,
)
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import (
    EpochIndex,
    EpochIndexEntry,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.pipeline.product_system.dependency_dag import (
    ProductBuildStatus,
    ProductStatus,
    build_toposorter,
    calculate_statuses,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
    ContinuumSubtractionKey,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_aperture_photometry import (
    dataframe_from_afrho_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_profile import (
    dataframe_from_afrho_from_radial_profile,
)


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config",
        nargs="?",
        help="Filename of project config",
        default="config.yaml",
    )

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def read_or_create_project_config(
    swift_project_config_path: pathlib.Path,
) -> CometProjectConfig | None:
    # check if project config exists, and offer to create if not
    if not swift_project_config_path.exists():
        print(
            f"Config file {swift_project_config_path} does not exist! Would you like to create one now? (y/n)"
        )
        return None
        # create_config = get_yes_no()
        # if create_config:
        #     create_swift_project_config_from_input(
        #         swift_project_config_path=swift_project_config_path
        #     )
        # else:
        #     return

    # load the project config
    swift_project_config = read_comet_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return None

    return swift_project_config


# # TODO: fix this function
# def create_swift_project_config_from_input(
#     swift_project_config_path: pathlib.Path,
# ) -> None:
#     """
#     Collect info on the data directories and how to identify the comet through JPL horizons,
#     and write it to a yaml config
#     """
#
#     swift_data_path = pathlib.Path(input("Directory of the downloaded swift data: "))
#
#     # try to validate that this path actually has data before accepting
#     test_of_swift_data = SwiftData(data_path=swift_data_path)
#     num_obsids = len(test_of_swift_data._find_all_observation_ids())
#     if num_obsids == 0:
#         rprint(
#             "There doesn't seem to be data in the necessary format at [blue]{swift_data_path}[/blue]!"
#         )
#     else:
#         rprint(
#             f"Found appropriate data with a total of [green]{num_obsids}[/green] observation IDs"
#         )
#
#     project_path = pathlib.Path(
#         input("Directory to store results and intermediate products: ")
#     )
#
#     jpl_horizons_id = input("JPL Horizons ID of the comet: ")
#
#     # TODO: this fails on invalid input, make it more robust
#     vm_quality = input(
#         f"Vectorial model quality {VectorialModelGridQuality.all_qualities()}: "
#     )
#
#     # TODO: this fails on invalid input, make it more robust
#     vm_backend = input(
#         f"Vectorial model backend {VectorialModelBackend.all_model_backends()}: "
#     )
#
#     # TODO: finish questions for vectorial_fitting_requires_km and near_far_split_radius_km
#
#     swift_project_config = SwiftProjectConfig(
#         swift_data_path=swift_data_path,
#         jpl_horizons_id=jpl_horizons_id,
#         project_path=project_path,
#         vectorial_model_quality=VectorialModelGridQuality(vm_quality),
#         vectorial_model_backend=VectorialModelBackend(vm_backend),
#         vectorial_fitting_requires_km=float(100_000),
#         near_far_split_radius_km=float(50_000),
#     )
#
#     print(f"Writing project config to {swift_project_config_path}...")
#     write_swift_project_config(
#         config_path=swift_project_config_path, swift_project_config=swift_project_config
#     )


def first_with_build_status(
    stat_dict: dict[ProductReference, ProductStatus], status: ProductBuildStatus
) -> ProductReference | None:

    return next(
        (ref for ref, stat in stat_dict.items() if stat.build_status == status),
        None,
    )


# def build_product_reference(
#     scp: Products, ref: ProductReference, verbose: bool = False, force: bool = False
# ) -> None:
#
#     if verbose:
#         print(f"Calculating dependencies for {ref.kind} --> {ref.key}")
#
#     if force:
#         print(f"Building {ref.kind} --> {ref.key}")
#         do_build(scp=scp, ref=ref)
#         return
#
#     ts = build_toposorter(scp=scp, target_product=ref)
#     stat_dict = calculate_statuses(scp=scp, ts=ts)
#
#     # show_pipeline_status_for_product(scp=scp, ref=ref)
#
#     first_ready = first_with_build_status(
#         stat_dict=stat_dict, status=ProductBuildStatus.ready
#     )
#     first_regen = first_with_build_status(
#         stat_dict=stat_dict, status=ProductBuildStatus.need_regen
#     )
#     first_stale = first_with_build_status(
#         stat_dict=stat_dict, status=ProductBuildStatus.stale
#     )
#
#     first_build = None
#     first_build = first_ready or first_regen or first_stale
#     if first_build is None:
#         print("Everything seems to be ready! Skipping build.")
#         print("")
#         return
#
#     if verbose:
#         print(f"Building {first_build.kind} --> {first_build.key}")
#     do_build(scp=scp, ref=first_build)
#     scp.regenerate()


def build_product_reference_loop(
    scp: Products, ref: ProductReference, verbose: bool = False
) -> None:

    if verbose:
        print(f"Calculating dependencies for {ref.kind} --> {ref.key}")

    while True:
        ts = build_toposorter(scp=scp, target_product=ref)
        stat_dict = calculate_statuses(scp=scp, ts=ts)

        show_pipeline_status_for_product(scp=scp, ref=ref)

        if stat_dict[ref].build_status == ProductBuildStatus.complete:
            # print(f"Product built!")
            break

        first_ready = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.ready
        )
        first_regen = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.need_regen
        )
        first_stale = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.stale
        )

        first_build = None
        first_build = first_ready or first_regen or first_stale
        if first_build is None:
            print("Everything seems to be ready! Skipping build.")
            print("")
            return

        print(f"Ready to build: {first_build}")
        do_build(scp=scp, ref=first_build)

        if ref.kind == ProductKind.epoch_index:
            # we need to do this after epoch_index to build out epoch subpipelines
            scp.regenerate()

        # wait_for_key()


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

    plot_images_multi(images=[fits_sum.data, fits_median.data], comet_centers=None)


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
        aapa_df.aperture_r_km, smoothed_total, label="smooth total", color="#688894"
    )
    plt.plot(
        aapa_df.aperture_r_km, smoothed_median, label="smooth median", color="#afac7c"
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

    plt.plot(xs, smoothed_ys, label="smoothed")
    plt.plot(crpfc.profile_axis_rs, crpfc.pixel_values, label="raw")
    # plt.plot(xs, crpfc.pixel_values / smoothed_ys, label="raw to smooth")

    plt.legend()
    plt.show()


def test_aperture_water_analysis_plotting(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    dust_redness: DustReddeningPercent,
) -> None:

    awp_prefs = enumerate_all_products_of(
        kind=ProductKind.aperture_water_production,
        epochs=[eid],
        oh_filters=[oh_filter],
        dust_filters=[dust_filter],
        stacking_methods=[StackingMethod.summation],
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

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.equivalent_q_h2o_sum + i * water_df.equivalent_q_h2o_sum_err,
    #         water_df.equivalent_q_h2o_sum - i * water_df.equivalent_q_h2o_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

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
    plt.ylim(
        1e26,
    )
    plt.title(f"OH filter: {str(oh_filter)}  Dust filter: {str(dust_filter)}")
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


def get_pipeline_status_for_product(
    scp: Products, ref: ProductReference
) -> ProductBuildStatus:

    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)
    return stat_dict[ref].build_status


def show_pipeline_status_for_product(
    scp: Products, ref: ProductReference, silent_if_complete: bool = True
) -> None:
    console = Console()
    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)
    for ref, stat in stat_dict.items():
        if stat.build_status == ProductBuildStatus.complete and silent_if_complete:
            continue
        console.print(ref)
        console.print(" -->  ", end="")
        console.print(stat)


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


def all_aperture_water_analysis(scp: Products, epoch_index: EpochIndex) -> None:

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
    for awp in incomplete_awp_prefs:
        assert isinstance(awp.key, ContinuumSubtractionKey)
        show_pipeline_status_for_product(scp=scp, ref=awp)
        build_product_reference_loop(scp=scp, ref=awp)
        # test_aperture_water_analysis_loading(scp=scp, ref=awp)


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


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)
    warnings.filterwarnings("ignore", category=VerifyWarning, append=True)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning, append=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 0)

    args = process_args()
    swift_comet_project_config_path = pathlib.Path(args.swift_project_config)

    comet_project_config = read_or_create_project_config(
        swift_project_config_path=swift_comet_project_config_path
    )

    if comet_project_config is None:
        print("Could not load a valid project configuration! Exiting.")
        return 1

    vectorial_model_settings_init(comet_project_config=comet_project_config)

    scp = Products(cfg=comet_project_config)

    build_all_data_ingestion_products(scp=scp)

    epoch_index = scp.load_epoch_index()
    assert epoch_index is not None

    # ---------

    # TODO: batch mode: stack all, background all, radial profile all
    # non-interactive
    stack_all_images(scp=scp, epoch_index=epoch_index)

    # interactive
    background_all_images(scp=scp, epoch_index=epoch_index)

    # non-interactive
    background_subtract_all_images(scp=scp, epoch_index=epoch_index)

    # TODO: aperture analyze all
    # radial profile all
    # aperture water production all
    # radial water production all

    # all_aperture_water_analysis(scp=scp, epoch_index=[epoch_index[7]])

    # ---------
    # TODO: Jorda 2008 empirical water production rates for V-band
    # TODO: function for selecting epoch by date --> return closest observation epoch index entry
    # selected_epoch = np.argmin(t - [x.observation_time for x in epoch_index])

    # TODO: water production builders need to convert the redness given in config to their own values for the filters being used
    # aperture is done - do the radial !!!!

    # TODO: manual water calculation: show aperture vs r for oh/dust filters and allow picking windows to average over for counts to use for continuum subtraction
    # Then either take a redness --> Q or use an expectation value for Q based on redness distribution
    # Since this has counts we can calculate mag/flux etc. manually as well

    # TODO: blue spot detection, Q expectation value
    # TODO: aperture photometry on asymmetric leftover after profile subtraction

    # TODO: latex table generation from epoch index
    # ---------

    eid = epoch_index[7]

    # ---------
    # Radial profile extraction testing
    pref = ProductReference(
        kind=ProductKind.radial_profile_from_cone,
        key=EpochSubpipelineKey(
            epoch_id=eid.epoch_id,
            filter_type=UvotFilter.ugrism,
            stacking_method=StackingMethod.summation,
        ),
    )
    # build_product_reference(scp=scp, ref=pref, verbose=True, force=True)
    build_product_reference_loop(scp=scp, ref=pref, verbose=True)
    # ---------

    # ---------
    # Aperture water analysis plotting
    # eid = epoch_index[4]
    test_aperture_water_analysis_plotting(
        scp=scp,
        eid=eid,
        oh_filter=UvotFilter.uw1,
        dust_filter=UvotFilter.uvv,
        dust_redness=DustReddeningPercent(30.0),
    )
    # test_aperture_water_analysis_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uw2,
    #     dust_filter=UvotFilter.uvv,
    #     dust_redness=DustReddeningPercent(30.0),
    # )
    # test_aperture_water_analysis_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uuu,
    #     dust_filter=UvotFilter.uvv,
    #     dust_redness=DustReddeningPercent(30.0),
    # )
    # # ---------

    # test_radial_water_production_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.ugrism,
    #     dust_filter=UvotFilter.uvv,
    #     dust_redness=DustReddeningPercent(40.0),
    # )

    # test_radial_water_production_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uw2,
    #     dust_filter=UvotFilter.uvv,
    #     dust_redness=DustReddeningPercent(30.0),
    # )
    # test_radial_water_production_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uuu,
    #     dust_filter=UvotFilter.uvv,
    #     dust_redness=DustReddeningPercent(30.0),
    # )
    # test_radial_water_production_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uw1,
    #     dust_filter=UvotFilter.uuu,
    #     dust_redness=DustReddeningPercent(30.0),
    # )

    # test_afrho_aperture_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uw1,
    #     dust_filter=UvotFilter.uuu,
    #     dust_redness=DustReddeningPercent(31.0),
    # )
    #
    # test_afrho_profile_plotting(
    #     scp=scp,
    #     eid=eid,
    #     oh_filter=UvotFilter.uw1,
    #     dust_filter=UvotFilter.uuu,
    #     dust_redness=DustReddeningPercent(31.0),
    # )

    # test_radial_profile_smoothing(scp=scp)

    # eid = epoch_index[11]
    # test_aperture_analysis_loading(scp=scp, eid=eid, filter_type=UvotFilter.uw1)
    # test_aperture_analysis_loading(scp=scp, eid=eid, filter_type=UvotFilter.uvv)

    # demo_reddening_recalculation()

    # test_background_result_loading(scp=scp)
    # test_fits_loading(scp=scp)
    # test_epoch_index_loading(scp=scp)
    # test_obs_log_metadata(scp=scp)


if __name__ == "__main__":
    sys.exit(main())
