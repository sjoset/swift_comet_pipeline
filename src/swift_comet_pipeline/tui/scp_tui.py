#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log
from argparse import ArgumentParser

import pandas as pd
from astropy.io.fits.card import VerifyWarning
from astropy.wcs.wcs import FITSFixedWarning
import astropy.units as u
from pandas.errors import SettingWithCopyWarning
from rich.console import Console

from swift_comet_pipeline.data_ingestion.epoch_index.build_epoch_index import (
    build_epoch_index,
)
from swift_comet_pipeline.data_ingestion.observation_log.build_observation_log import (
    build_observation_log,
)
from swift_comet_pipeline.data_ingestion.observation_log.slice_observation_log_into_epochs import (
    add_epoch_ids_by_time_window,
)
from swift_comet_pipeline.data_ingestion.orbit_data.find_perihelion import (
    find_perihelia,
)
from swift_comet_pipeline.data_ingestion.orbit_data.orbit_data_download import (
    orbit_data_download,
)
from swift_comet_pipeline.data_ingestion.veto.gui_manual_veto import manual_veto
from swift_comet_pipeline.image_manipulation.utility.plot_image_multi import (
    plot_images_multi,
)
from swift_comet_pipeline.product_system.dependency_dag import (
    ProductBuildStatus,
    ProductStatus,
    build_toposorter,
    calculate_statuses,
)
from swift_comet_pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    GlobalKey,
    ProductKind,
    ProductReference,
    Products,
)
from swift_comet_pipeline.project_configuration.read_swift_comet_project_config import (
    read_swift_comet_project_config,
)
from swift_comet_pipeline.scp_types.compound.swift_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter
from swift_comet_pipeline.stacking.stacking import do_stacking
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.ui.mpl_ui.mpl_ui_observation_log_slicing import (
    gui_select_epoch_time_window,
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


def do_observation_log_raw(scp: Products) -> None:

    print("calling observation log raw build")
    swift_data = SwiftData(data_path=scp.cfg.swift_data_path)
    obs_log_df = build_observation_log(
        swift_data=swift_data, horizons_id=scp.cfg.jpl_horizons_id
    )

    if obs_log_df is None:
        print("Error while building observation log!  Not saving.")
        return

    scp.save_raw_log(df=obs_log_df)


def do_epoch_identification(scp: Products) -> None:

    print("Slicing epochs..")
    obs_log_df = scp.load_raw_log()
    assert obs_log_df is not None

    dt = gui_select_epoch_time_window(obs_log=obs_log_df)
    df = add_epoch_ids_by_time_window(obs_log=obs_log_df, max_time_between_obs=dt)
    df.attrs = {"time_slice_hours": str(dt.to_value(u.hour))}  # type: ignore

    scp.save_epoch_log(df=df)


def do_image_veto(scp: Products) -> None:

    print("Starting veto..")
    epoch_log_df = scp.load_epoch_log()
    assert epoch_log_df is not None

    # call veto
    df = manual_veto(scp=scp)

    # scp.save_obs_log(df=df)


def do_earth_orbit_download(scp: Products) -> None:

    print("Downloading earth orbit data..")
    earth_df = orbit_data_download(scp=scp, horizons_id=None)

    scp.save_earth_orbit_data(df=earth_df)


def do_comet_orbit_download(scp: Products) -> None:

    print("Downloading comet orbit data..")
    comet_df = orbit_data_download(scp=scp, horizons_id=scp.cfg.jpl_horizons_id)

    scp.save_comet_orbit_data(df=comet_df)


def do_epoch_index(scp: Products) -> None:

    print("Building epoch index...")

    p_list = find_perihelia(scp=scp)
    if p_list is None:
        return
    # for p in p_list:
    #     print(f"T_p: {p.t_perihelion.utc}\t{p.rh_au} AU")

    epoch_index = build_epoch_index(scp=scp)
    if epoch_index is None:
        return
    scp.save_epoch_index(epoch_index=epoch_index)


def do_stack(scp: Products, ref: ProductReference) -> None:

    print(f"Building stack for {ref} ...")

    assert isinstance(ref.key, EpochSubpipelineKey)
    sum_med_exp = do_stacking(scp=scp, key=ref.key)

    if sum_med_exp is None:
        print("Stacking failed!")
        return

    img_sum, img_median, img_exposure_map = sum_med_exp

    # TODO: show the images and ask whether we want to save or not
    sum_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.summation,
        ),
    )
    med_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.median,
        ),
    )
    exposure_sum_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.summation,
        ),
    )
    exposure_med_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.median,
        ),
    )
    scp.save_fits_image(img=img_sum, ref=sum_ref)
    scp.save_fits_image(img=img_median, ref=med_ref)
    scp.save_fits_image(img=img_exposure_map, ref=exposure_sum_ref)
    scp.save_fits_image(img=img_exposure_map, ref=exposure_med_ref)

    print(f"Done stacking!")


def do_build(scp: Products, target_product: ProductReference) -> None:

    if target_product == ProductReference(
        kind=ProductKind.observation_log_raw, key=GlobalKey()
    ):
        do_observation_log_raw(scp=scp)

    if target_product == ProductReference(
        kind=ProductKind.observation_log_with_epochs, key=GlobalKey()
    ):
        do_epoch_identification(scp=scp)

    if target_product == ProductReference(
        kind=ProductKind.observation_log_with_vetoes, key=GlobalKey()
    ):
        do_image_veto(scp=scp)

    if target_product == ProductReference(
        kind=ProductKind.orbit_data_earth, key=GlobalKey()
    ):
        do_earth_orbit_download(scp=scp)

    if target_product == ProductReference(
        kind=ProductKind.orbit_data_comet, key=GlobalKey()
    ):
        do_comet_orbit_download(scp=scp)

    if target_product == ProductReference(ProductKind.epoch_index, key=GlobalKey()):
        do_epoch_index(scp=scp)

    if (
        target_product.kind == ProductKind.stacked_image_with_background
        or target_product.kind == ProductKind.stacked_image_exposure_map
    ):
        do_stack(scp=scp, ref=target_product)


def first_with_build_status(
    stat_dict: dict[ProductReference, ProductStatus], status: ProductBuildStatus
) -> ProductReference | None:

    return next(
        (ref for ref, stat in stat_dict.items() if stat.build_status == status),
        None,
    )


def build_target_product(scp: Products, target_product: ProductReference) -> None:

    console = Console()
    ts = build_toposorter(scp=scp, target_product=target_product)
    stat_dict = calculate_statuses(scp=scp, ts=ts)
    print("")
    print("------- build status ---------")
    for ref, stat in stat_dict.items():
        console.print(f"{ref} --> ", end="")
        console.print(stat)
    #     # console.print(scp.path_for(ref=ref))
    #     # console.print()

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
    do_build(scp=scp, target_product=first_build)
    scp.regenerate()
    # TODO: instead of one-shot, loop until we are done


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


def test_fits_loading(scp: Products) -> None:

    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    fits_sum = scp.load_fits_image(ref=target_ref)
    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.median,
        ),
    )
    fits_median = scp.load_fits_image(ref=target_ref)

    if fits_sum is None or fits_median is None:
        print(f"fits image of {target_ref} failed to load!")
        return

    plot_images_multi(images=[fits_sum.data, fits_median.data], comet_centers=None)
    # print(fits_sum.header)


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

    scp = Products(cfg=comet_project_config)

    print("Checking data ingestion ...")
    target_ref = ProductReference(kind=ProductKind.epoch_index, key=GlobalKey())
    build_target_product(scp=scp, target_product=target_ref)
    show_pipeline_status_for_product(scp=scp, ref=target_ref)

    print("Checking sum stacks")
    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )

    build_target_product(scp=scp, target_product=target_ref)
    show_pipeline_status_for_product(scp=scp, ref=target_ref)

    print("Checking median stacks")
    target_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.median,
        ),
    )

    build_target_product(scp=scp, target_product=target_ref)
    show_pipeline_status_for_product(scp=scp, ref=target_ref)

    print("Checking exposure maps")
    target_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )

    build_target_product(scp=scp, target_product=target_ref)
    show_pipeline_status_for_product(scp=scp, ref=target_ref)

    target_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id="000_2014_Aug_14",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.median,
        ),
    )

    build_target_product(scp=scp, target_product=target_ref)
    show_pipeline_status_for_product(scp=scp, ref=target_ref)

    # test_fits_loading(scp=scp)
    # test_epoch_index_loading(scp=scp)
    # test_obs_log_metadata(scp=scp)


if __name__ == "__main__":
    sys.exit(main())
