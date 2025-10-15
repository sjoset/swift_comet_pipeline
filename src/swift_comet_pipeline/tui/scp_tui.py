#!/usr/bin/env python3

from dataclasses import replace
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

from swift_comet_pipeline.image_manipulation.utility.plot_image_multi import (
    plot_images_multi,
)
from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
)
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.builders.aperture_photometry_analyzer import (
    do_aperture_photometry_analysis,
)
from swift_comet_pipeline.builders.aperture_water_production_calculator import (
    do_aperture_water_production,
)
from swift_comet_pipeline.builders.background_determiner import (
    do_background_determination,
)
from swift_comet_pipeline.builders.background_subtractor import (
    do_background_subtraction,
)
from swift_comet_pipeline.builders.epoch_identifier import do_epoch_identification
from swift_comet_pipeline.builders.epoch_indexer import do_epoch_index
from swift_comet_pipeline.builders.image_vetoer import do_image_veto
from swift_comet_pipeline.builders.orbit_downloader import (
    do_comet_orbit_download,
    do_earth_orbit_download,
)
from swift_comet_pipeline.builders.radial_profile_extractor import (
    do_radial_profile_from_cone,
)
from swift_comet_pipeline.builders.raw_observation_log_builder import (
    do_observation_log_raw,
)
from swift_comet_pipeline.builders.stacker import do_stack
from swift_comet_pipeline.pipeline.product_system.dependency_dag import (
    ProductBuildStatus,
    ProductStatus,
    build_toposorter,
    calculate_statuses,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    GlobalKey,
    ProductKind,
    ProductReference,
    Products,
    WaterProductionKey,
    add_water_product_products_to_registry,
)
from swift_comet_pipeline.pipeline.project_configuration.read_swift_comet_project_config import (
    read_swift_comet_project_config,
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
    swift_project_config = read_swift_comet_project_config(swift_project_config_path)
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


def do_build(scp: Products, ref: ProductReference) -> None:

    if ref == ProductReference(kind=ProductKind.observation_log_raw, key=GlobalKey()):
        do_observation_log_raw(scp=scp)

    if ref == ProductReference(
        kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
    ):
        do_epoch_identification(scp=scp)

    if ref == ProductReference(
        kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
    ):
        do_image_veto(scp=scp)

    if ref == ProductReference(kind=ProductKind.orbit_data_earth, key=GlobalKey()):
        do_earth_orbit_download(scp=scp)

    if ref == ProductReference(kind=ProductKind.orbit_data_comet, key=GlobalKey()):
        do_comet_orbit_download(scp=scp)

    if ref == ProductReference(ProductKind.epoch_index, key=GlobalKey()):
        do_epoch_index(scp=scp)

    if (
        ref.kind == ProductKind.stacked_image_with_background
        or ref.kind == ProductKind.stacked_image_exposure_map
    ):
        do_stack(scp=scp, ref=ref)

    if ref.kind == ProductKind.background_determination:
        do_background_determination(scp=scp, ref=ref)

    if ref.kind == ProductKind.bg_subtracted_stacked_image:
        do_background_subtraction(scp=scp, ref=ref)

    if ref.kind == ProductKind.annular_aperture_photometry_analysis:
        do_aperture_photometry_analysis(scp=scp, ref=ref)

    if ref.kind == ProductKind.radial_profile_from_cone:
        do_radial_profile_from_cone(scp=scp, ref=ref)

    if ref.kind == ProductKind.aperture_water_production:
        do_aperture_water_production(scp=scp, ref=ref)


def first_with_build_status(
    stat_dict: dict[ProductReference, ProductStatus], status: ProductBuildStatus
) -> ProductReference | None:

    return next(
        (ref for ref, stat in stat_dict.items() if stat.build_status == status),
        None,
    )


def build_product_reference(scp: Products, ref: ProductReference) -> None:

    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)

    show_pipeline_status_for_product(scp=scp, ref=ref)

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

    print("")
    print(f"Ready to build: {first_build}")
    do_build(scp=scp, ref=first_build)
    scp.regenerate()
    # TODO: instead of one-shot, loop until we are done


def build_product_reference_loop(scp: Products, ref: ProductReference) -> None:

    # console = Console()

    while True:
        ts = build_toposorter(scp=scp, target_product=ref)
        stat_dict = calculate_statuses(scp=scp, ts=ts)
        # print("")
        # print("------- build status ---------")
        # for ref, stat in stat_dict.items():
        #     console.print(f"{ref} --> ", end="")
        #     console.print(stat)

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
        # print(epoch)
        print(
            epoch.epoch_id,
            "\t",
            epoch.epoch_length,
            "\t",
            epoch.time_from_perihelion,
            "\t",
            epoch.observation_time,
        )

    # epoch_index_sorted = sorted(epoch_index, key=lambda x: x.epoch_id, reverse=True)
    # print(epoch_index_sorted)


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


def test_aperture_analysis_loading(scp: Products) -> None:
    target_ref = ProductReference(
        kind=ProductKind.annular_aperture_photometry_analysis,
        key=EpochSubpipelineKey(
            # epoch_id="000_2014_Aug_14",
            # epoch_id="001_2014_Nov_05",
            epoch_id="003_2015_Apr_28",
            # epoch_id="005_2015_Aug_11",
            # epoch_id="008_2016_Mar_14",
            # epoch_id="009_2016_Apr_10",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    pkey = target_ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    aaa_df = scp.load_annular_aperture_analysis(key=pkey)
    assert aaa_df is not None

    smoothed_total = savgol_filter(aaa_df.cumulative_sum, window_length=20, polyorder=2)
    smoothed_median = savgol_filter(
        aaa_df.cumulative_median_signal, window_length=20, polyorder=2
    )

    # plt.scatter(x=aaa_df.aperture_r_km, y=aaa_df.cumulative_sum, color="#688894")
    # plt.scatter(
    #     x=aaa_df.aperture_r_km, y=aaa_df.cumulative_median_signal, color="#688894"
    # )

    plt.plot(
        aaa_df.aperture_r_km, smoothed_total, label="smooth total", color="#688894"
    )
    plt.plot(
        aaa_df.aperture_r_km, smoothed_median, label="smooth median", color="#afac7c"
    )

    # plt.errorbar(
    #     aaa_df.aperture_r_km,
    #     aaa_df.cumulative_sum,
    #     yerr=aaa_df.cumulative_sum_err,
    #     label="total",
    # )
    #
    # plt.errorbar(
    #     aaa_df.aperture_r_km,
    #     aaa_df.cumulative_median_signal,
    #     yerr=aaa_df.cumulative_median_err,
    #     label="total median",
    # )

    for i in range(4):
        plt.fill_between(
            aaa_df.aperture_r_km,
            aaa_df.cumulative_sum - i * aaa_df.cumulative_sum_err,
            aaa_df.cumulative_sum + i * aaa_df.cumulative_sum_err,
            alpha=0.2,
            color="#688894",
        )
        plt.fill_between(
            aaa_df.aperture_r_km,
            aaa_df.cumulative_median_signal - i * aaa_df.cumulative_median_err,
            aaa_df.cumulative_median_signal + i * aaa_df.cumulative_median_err,
            alpha=0.2,
            color="#afac7c",
        )
    plt.legend()
    plt.xlim(0, 600000)
    plt.ylim(
        0,
    )
    plt.show()


def test_radial_profile_loading(scp: Products) -> None:

    target_ref = ProductReference(
        kind=ProductKind.annular_aperture_photometry_analysis,
        key=EpochSubpipelineKey(
            # epoch_id="003_2015_Apr_28",
            epoch_id="004_2015_Jun_19",
            # epoch_id="005_2015_Aug_11",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    pkey = target_ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    crpfc = scp.load_extracted_radial_profile(key=pkey)

    xs = crpfc.profile_axis_rs
    ys = crpfc.pixel_values

    smoothed_ys = savgol_filter(ys, window_length=10, polyorder=2)

    plt.plot(xs, smoothed_ys, label="smoothed")
    plt.plot(crpfc.profile_axis_rs, crpfc.pixel_values, label="raw")
    # plt.plot(xs, crpfc.pixel_values / smoothed_ys, label="raw to smooth")
    plt.legend()
    plt.show()

    # plt.show()


def show_pipeline_status_for_product(scp: Products, ref: ProductReference) -> None:
    console = Console()
    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)
    for ref, stat in stat_dict.items():
        console.print(f"{ref} --> ", end="")
        console.print(stat)
        # console.print(scp.path_for(ref=ref))
        # console.print()


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

    # print("Checking data ingestion ...")
    # epoch_index_ref = ProductReference(kind=ProductKind.epoch_index, key=GlobalKey())
    # build_target_product_loop(scp=scp, target_product=epoch_index_ref)
    # show_pipeline_status_for_product(scp=scp, ref=target_ref)

    # print("Checking sum stacks")
    # target_ref = ProductReference(
    #     kind=ProductKind.stacked_image_with_background,
    #     key=EpochSubpipelineKey(
    #         epoch_id="000_2014_Aug_14",
    #         filter_type=UvotFilter.uw1,
    #         stacking_method=StackingMethod.summation,
    #     ),
    # )
    #
    # build_target_product(scp=scp, target_product=target_ref)
    # show_pipeline_status_for_product(scp=scp, ref=target_ref)
    #
    # print("Checking median stacks")
    # target_ref = ProductReference(
    #     kind=ProductKind.stacked_image_with_background,
    #     key=EpochSubpipelineKey(
    #         epoch_id="000_2014_Aug_14",
    #         filter_type=UvotFilter.uw1,
    #         stacking_method=StackingMethod.median,
    #     ),
    # )
    #
    # build_target_product(scp=scp, target_product=target_ref)
    # show_pipeline_status_for_product(scp=scp, ref=target_ref)

    # print("Checking exposure maps")
    # target_ref = ProductReference(
    #     kind=ProductKind.stacked_image_exposure_map,
    #     key=EpochSubpipelineKey(
    #         epoch_id="000_2014_Aug_14",
    #         filter_type=UvotFilter.uw1,
    #         stacking_method=StackingMethod.summation,
    #     ),
    # )

    # print("Checking background determinations")
    # target_ref = ProductReference(
    #     kind=ProductKind.background_determination,
    #     key=EpochSubpipelineKey(
    #         # epoch_id="000_2014_Aug_14",
    #         # epoch_id="011_2016_Nov_24",
    #         # epoch_id="003_2015_Apr_28",
    #         epoch_id="004_2015_Jun_19",
    #         filter_type=UvotFilter.uw1,
    #         stacking_method=StackingMethod.summation,
    #     ),
    # )
    #
    # build_target_product(scp=scp, target_product=target_ref)
    # show_pipeline_status_for_product(scp=scp, ref=target_ref)

    # target_ref = ProductReference(
    #     kind=ProductKind.bg_subtracted_stacked_image,
    #     key=EpochSubpipelineKey(
    #         epoch_id="000_2014_Aug_14",
    #         filter_type=UvotFilter.uw1,
    #         stacking_method=StackingMethod.median,
    #     ),
    # )

    # pkey = EpochSubpipelineKey(
    #     # epoch_id="000_2014_Aug_14",
    #     # epoch_id="001_2014_Nov_05",
    #     # epoch_id="003_2015_Apr_28",
    #     # epoch_id="005_2015_Aug_11",
    #     epoch_id="008_2016_Mar_14",
    #     # epoch_id="009_2016_Apr_10",
    #     filter_type=UvotFilter.uw1,
    #     stacking_method=StackingMethod.summation,
    # )
    #
    # target_ref = ProductReference(
    #     kind=ProductKind.annular_aperture_photometry_analysis, key=pkey
    # )
    # build_product_reference_loop(scp=scp, ref=target_ref)

    # target_ref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=pkey)
    # build_product_reference_loop(scp=scp, ref=target_ref)

    # show_pipeline_status_for_product(scp=scp, ref=target_ref)

    # TODO: all builders should look for existing products and attempt to use those same parameters to re-build

    # TODO: add afrho product for each requested dust filter
    # TODO: test uuu filter in aperture/water analysis

    epoch_id = "000_2014_Aug_14"
    # epoch_id = "001_2014_Nov_05"
    # epoch_id = "002_2014_Dec_20"
    # epoch_id = "003_2015_Apr_28"
    # epoch_id = "004_2015_Jun_19"
    # epoch_id = "005_2015_Aug_11"
    # epoch_id = "008_2016_Mar_14"
    # epoch_id = "009_2016_Apr_10"

    pkey_uw1 = EpochSubpipelineKey(
        epoch_id=epoch_id,
        filter_type=UvotFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    pkey_uvv = replace(pkey_uw1, filter_type=UvotFilter.uvv)
    for key in [pkey_uvv, pkey_uw1]:
        p_ref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=key)
        build_product_reference_loop(scp=scp, ref=p_ref)

    oh_filters = [UvotFilter.uw1, UvotFilter.uw2]
    dust_filters = [UvotFilter.uvv, UvotFilter.uuu]
    epoch_index = scp.load_epoch_index()
    assert epoch_index is not None
    add_water_product_products_to_registry(
        reg=scp.reg,
        epoch_index=epoch_index,
        oh_filters=oh_filters,
        dust_filters=dust_filters,
    )

    wkey = WaterProductionKey(
        epoch_id=epoch_id,
        oh_filter=UvotFilter.uw1,
        dust_filter=UvotFilter.uvv,
        stacking_method=StackingMethod.summation,
    )
    ap_wat_ref = ProductReference(kind=ProductKind.aperture_water_production, key=wkey)

    build_product_reference(scp=scp, ref=ap_wat_ref)
    show_pipeline_status_for_product(scp=scp, ref=ap_wat_ref)

    # test_radial_profile_loading(scp=scp)
    # test_aperture_analysis_loading(scp=scp)
    # test_background_result_loading(scp=scp)
    # test_fits_loading(scp=scp)
    # test_epoch_index_loading(scp=scp)
    # test_obs_log_metadata(scp=scp)


if __name__ == "__main__":
    sys.exit(main())
