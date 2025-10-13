#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log
from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from astropy.io.fits.card import VerifyWarning
from astropy.wcs.wcs import FITSFixedWarning
import astropy.units as u
from pandas.errors import SettingWithCopyWarning
from rich.console import Console
from tqdm import tqdm
import matplotlib.pyplot as plt

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
from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.image_manipulation.utility.plot_image_multi import (
    plot_images_multi,
)
from swift_comet_pipeline.photometry.aperture.aperture_count_rate import (
    aperture_analysis,
)
from swift_comet_pipeline.photometry.aperture.concentric_annuli import (
    make_concentric_annular_apertures,
)
from swift_comet_pipeline.photometry.background.determine_background import (
    determine_background,
)
from swift_comet_pipeline.photometry.comet.radial_profile_from_cone_ui import (
    profile_extraction_from_cone,
)
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
from swift_comet_pipeline.scp_types.compound.annular_aperture_profile import (
    AnnularApertureProfileEntry,
    dataframe_from_annular_aperture_profile,
)
from swift_comet_pipeline.scp_types.compound.swift_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive.aperture_count_rate_analysis import (
    aperture_count_rate_analysis_kwargs,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.scp_types.primitive.swift_uvot_image import SwiftUvotImage
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
    manual_veto(scp=scp)


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


def do_background_determination(scp: Products, ref: ProductReference) -> None:

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    stack_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background, key=pkey
    )
    stack_fits = scp.load_fits_image(ref=stack_ref)
    if not stack_fits:
        print(f"Could not load image for product {stack_ref}!  Skipping.")
        return
    stack_img = stack_fits.data
    assert isinstance(stack_img, SwiftUvotImage)

    exp_ref = ProductReference(kind=ProductKind.stacked_image_exposure_map, key=pkey)
    exp_fits = scp.load_fits_image(ref=exp_ref)
    if not exp_fits:
        print(f"Could not exposure map for {exp_ref}!  Skipping.")
        return
    exp_map = exp_fits.data
    assert isinstance(exp_map, SwiftUvotImage)

    bg_result = determine_background(
        img=stack_img,
        exposure_map=exp_map,
        filter_type=pkey.filter_type,
        epoch_id=pkey.epoch_id,
    )

    if not bg_result:
        print(f"Could not get background result for {ref}!")
        return

    scp.save_background_result(bg_result=bg_result, key=pkey)


def do_background_subtraction(scp: Products, ref: ProductReference) -> None:

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    stack_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background, key=pkey
    )
    stack_fits = scp.load_fits_image(ref=stack_ref)
    if stack_fits is None:
        print(f"Could not load image for product {stack_ref}!  Skipping.")
        return
    stack_img = stack_fits.data
    assert isinstance(stack_img, SwiftUvotImage)

    bgr = scp.load_background_result(key=pkey)
    if bgr is None:
        print(
            f"Could not load the background determination for {stack_ref}!  Skipping."
        )
        return

    bg_sub_fits = stack_fits.copy()
    bg_sub_fits.data = stack_img - bgr.b_hat
    bg_sub_fits.header["bg_subtracted"] = True

    bg_sub_ref = ProductReference(
        kind=ProductKind.bg_subtracted_stacked_image, key=pkey
    )
    scp.save_fits_image(img=bg_sub_fits, ref=bg_sub_ref)


def do_aperture_photometry_analysis(scp: Products, ref: ProductReference) -> None:

    # parameters of analysis
    max_aperture_radius = 8e5 * u.km  # type: ignore
    num_concentric_apertures = 400

    # load epoch info and the image to be analyzed
    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    eid = scp.load_epoch_index_entry(epoch_id=pkey.epoch_id)
    assert eid is not None
    img_ref = ProductReference(kind=ProductKind.bg_subtracted_stacked_image, key=pkey)
    img_fits = scp.load_fits_image(ref=img_ref)
    assert img_fits is not None
    img = img_fits.data
    assert isinstance(img, SwiftUvotImage)
    bg_result = scp.load_background_result(key=pkey)
    assert bg_result is not None

    # derived quantities & information we need
    r_max_pix = max_aperture_radius.to_value(u.km) / eid.km_per_pix  # type: ignore
    comet_center = get_uvot_image_center(img=img)

    # make annular apertures with a circle aperture at r=0
    annular_apertures = make_concentric_annular_apertures(
        ap_center=comet_center,
        min_radius=0.0,
        max_radius=r_max_pix,
        num_concentric_apertures=num_concentric_apertures,
    )
    aperture_r_pix = [float(annular_apertures[0].r)] + [  # type: ignore
        float(x.r_out) for x in annular_apertures[1:]  # type: ignore
    ]
    aperture_dr_pix = np.array([annular_apertures[0].r]) + np.diff(aperture_r_pix)  # type: ignore

    aperture_r_km = [x * eid.km_per_pix for x in aperture_r_pix]
    aperture_dr_km = [x * eid.km_per_pix for x in aperture_dr_pix]

    assert eid.exposure_times.get(pkey.filter_type, None) is not None

    print("Starting counts ...")
    annular_analyses = [
        aperture_analysis(
            img=img,
            ap=ap,
            background=bg_result,
            exposure_time_s=eid.exposure_times[pkey.filter_type],
        )
        for ap in tqdm(annular_apertures)
    ]

    # magnitudes_list = [
    #     magnitude_from_count_rate(
    #         count_rate=CountRate(
    #             value=x.median_count_rate, sigma=np.sqrt(x.total_count_rate_variance)
    #         ),
    #         filter_type=pkey.filter_type,
    #     )
    #     for x in annular_analyses
    # ]
    # magnitude = [x.value for x in magnitudes_list]
    # magnitude_err = [x.sigma for x in magnitudes_list]

    ap_profile_data = {
        "aperture_r_pix": aperture_r_pix,
        "aperture_r_km": aperture_r_km,
        "aperture_dr_pix": aperture_dr_pix,
        "aperture_dr_km": aperture_dr_km,
        # "magnitude": magnitude,
        # "magnitude_err": magnitude_err,
    }
    ap_profile_data_names = list(ap_profile_data.keys())
    ap_profile_data_lists = [ap_profile_data[name] for name in ap_profile_data_names]

    annular_aperture_profile = [
        AnnularApertureProfileEntry(
            **aperture_count_rate_analysis_kwargs(aa),
            **dict(zip(ap_profile_data_names, data_lists)),
        )
        for aa, *data_lists in zip(annular_analyses, *ap_profile_data_lists)
    ]

    analysis_metadata = {
        "max_aperture_radius_km": str(max_aperture_radius.to_value(u.km)),  # type: ignore
        "num_concentric_apertures": str(num_concentric_apertures),
    }
    annular_aperture_analysis_df = dataframe_from_annular_aperture_profile(
        annular_aperture_profile=annular_aperture_profile
    )
    annular_aperture_analysis_df.attrs = analysis_metadata  # type: ignore

    scp.save_annular_aperture_analysis(df=annular_aperture_analysis_df, key=pkey)


def do_radial_profile_from_cone(scp: Products, ref: ProductReference) -> None:
    crpfc = profile_extraction_from_cone(scp=scp, ref=ref)

    assert isinstance(ref.key, EpochSubpipelineKey)
    # print(
    #     f"Saving profile to {scp.save_extracted_radial_profile(crp=crpfc, key=ref.key)}"
    # )
    scp.save_extracted_radial_profile(crp=crpfc, key=ref.key)


def do_aperture_water_production(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, WaterProductionKey)

    oh_subpipe_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.oh_filter,
        stacking_method=ref.key.stacking_method,
    )
    dust_subpipe_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
    )

    # oh_ref = ProductReference(
    #     kind=ProductKind.annular_aperture_photometry_analysis,
    #     key=oh_subpipe_key,
    # )
    # dust_ref = ProductReference(
    #     kind=ProductKind.annular_aperture_photometry_analysis, key=dust_subpipe_key
    # )

    oh_aa = scp.load_annular_aperture_analysis(key=oh_subpipe_key)
    dust_aa = scp.load_annular_aperture_analysis(key=dust_subpipe_key)

    print(oh_subpipe_key, dust_subpipe_key)

    print("done")


def do_build(scp: Products, ref: ProductReference) -> None:

    # print(f"do_build:")
    # print("----------")
    # print(f"Building: {ref.kind}")
    # print(f"Key: {ref.key}")

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
            # epoch_id="003_2015_Apr_28",
            # epoch_id="005_2015_Aug_11",
            epoch_id="008_2016_Mar_14",
            # epoch_id="009_2016_Apr_10",
            filter_type=UvotFilter.uw1,
            stacking_method=StackingMethod.summation,
        ),
    )
    pkey = target_ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    aaa_df = scp.load_annular_aperture_analysis(key=pkey)
    assert aaa_df is not None

    aaa_df["total_aperture_area"] = aaa_df.ap_num_pixels.cumsum()

    aaa_df["median_signal"] = aaa_df.median_count_rate * aaa_df.ap_num_pixels
    aaa_df["total_median_signal"] = aaa_df.median_signal.cumsum()
    # aaa_df["total_median_variance"] = aaa_df.median_va
    # aaa_df["median_snr"] = aaa_df.total_median_signal / aaa_df.total_signal_variance

    aaa_df["total_signal"] = aaa_df.total_count_rate.cumsum()
    aaa_df["total_signal_variance"] = aaa_df.total_count_rate_variance.cumsum()
    aaa_df["total_signal_err"] = np.sqrt(aaa_df.total_signal_variance)
    aaa_df["total_signal_snr"] = aaa_df.total_signal / aaa_df.total_signal_err
    # print(aaa_df)
    print(f"Aperture analysis metadata: {aaa_df.attrs}")

    smoothed_total = savgol_filter(aaa_df.total_signal, window_length=20, polyorder=2)
    smoothed_median = savgol_filter(
        aaa_df.total_median_signal, window_length=20, polyorder=2
    )
    # aaa_df.plot(kind="scatter", x="aperture_r_km", y="total_median_signal")
    # aaa_df.plot(kind="scatter", x="aperture_r_km", y="total_signal")
    plt.plot(
        aaa_df.aperture_r_km, smoothed_total, label="smooth total", color="#688894"
    )
    plt.plot(
        aaa_df.aperture_r_km, smoothed_median, label="smooth median", color="#afac7c"
    )
    # plt.errorbar(
    #     aaa_df.aperture_r_km,
    #     aaa_df.total_signal,
    #     yerr=aaa_df.total_signal_err,
    #     label="total",
    # )
    # plt.errorbar(
    #     aaa_df.aperture_r_km,
    #     aaa_df.total_median_signal,
    #     yerr=aaa_df.total_signal_err,
    #     label="total median",
    # )
    for i in range(4):
        plt.fill_between(
            aaa_df.aperture_r_km,
            aaa_df.total_signal - i * aaa_df.total_signal_err,
            aaa_df.total_signal + i * aaa_df.total_signal_err,
            alpha=0.2,
            color="#688894",
        )
        plt.fill_between(
            aaa_df.aperture_r_km,
            aaa_df.total_median_signal - i * aaa_df.total_signal_err,
            aaa_df.total_median_signal + i * aaa_df.total_signal_err,
            alpha=0.2,
            color="#afac7c",
        )
    plt.legend()
    plt.show()


def test_radial_profile_loading(scp: Products) -> None:

    target_ref = ProductReference(
        kind=ProductKind.annular_aperture_photometry_analysis,
        key=EpochSubpipelineKey(
            # epoch_id="003_2015_Apr_28",
            epoch_id="005_2015_Aug_11",
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
    plt.plot(xs, crpfc.pixel_values / smoothed_ys, label="raw to smooth")
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
    # build_target_product_loop(scp=scp, target_product=target_ref)
    #
    # target_ref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=pkey)
    # build_target_product_loop(scp=scp, target_product=target_ref)
    #
    # show_pipeline_status_for_product(scp=scp, ref=target_ref)

    epoch_id = "008_2016_Mar_14"
    # epoch_id = "009_2016_Apr_10"
    # epoch_id = "000_2014_Aug_14"
    # epoch_id = "003_2015_Apr_28"
    # epoch_id = "005_2015_Aug_11"

    # pkey_uw1 = EpochSubpipelineKey(
    #     epoch_id=epoch_id,
    #     filter_type=UvotFilter.uw1,
    #     stacking_method=StackingMethod.summation,
    # )
    # pkey_uvv = replace(pkey_uw1, filter_type=UvotFilter.uvv)
    #
    # for key in [pkey_uvv, pkey_uw1]:
    #     p_ref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=key)
    #     build_target_product_loop(scp=scp, target_product=p_ref)

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
    # show_pipeline_status_for_product(scp=scp, ref=ap_wat_ref)

    # test_radial_profile_loading(scp=scp)
    test_aperture_analysis_loading(scp=scp)
    # test_background_result_loading(scp=scp)
    # test_fits_loading(scp=scp)
    # test_epoch_index_loading(scp=scp)
    # test_obs_log_metadata(scp=scp)


if __name__ == "__main__":
    sys.exit(main())
