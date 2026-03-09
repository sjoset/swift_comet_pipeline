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
from pandas.errors import SettingWithCopyWarning

from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.project_configuration.read_comet_project_config import (
    read_comet_project_config,
)
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.builders.batch_building import (
    all_afrho_from_apertures,
    all_afrho_from_radial_profiles,
    all_aperture_analysis,
    all_aperture_water_analysis,
    all_bayesian_blue_spot_lightcurves,
    all_bayesian_water_production_lightcurves,
    all_blue_spot_lightcurves,
    all_radial_profile_extraction,
    all_radial_profile_subtracted_images,
    all_radial_profile_water_analysis,
    all_water_production_lightcurves,
    background_all_images,
    background_subtract_all_images,
    build_all_data_ingestion_products,
    stack_all_images,
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

    # TODO: EpochIndexEntry should have rh_au tagged with negative for pre-perihelion

    # non-interactive
    stack_all_images(scp=scp, epoch_index=epoch_index)

    # interactive
    background_all_images(scp=scp, epoch_index=epoch_index)

    # non-interactive
    background_subtract_all_images(scp=scp, epoch_index=epoch_index)

    # interactive
    all_radial_profile_extraction(scp=scp, epoch_index=epoch_index)

    # non-interactive
    all_radial_profile_subtracted_images(scp=scp, epoch_index=epoch_index)

    all_aperture_analysis(scp=scp, epoch_index=epoch_index)

    all_afrho_from_apertures(scp=scp, epoch_index=epoch_index)

    all_radial_profile_water_analysis(scp=scp, epoch_index=epoch_index)

    all_afrho_from_radial_profiles(scp=scp, epoch_index=epoch_index)

    # all_aperture_water_analysis(scp=scp, epoch_index=epoch_index, n_jobs=4)
    all_aperture_water_analysis(scp=scp, epoch_index=epoch_index)

    # ---------
    # TODO: Jorda 2008 empirical water production rates for V-band
    # TODO: function for selecting epoch by date --> return closest observation epoch index entry
    # epoch_i = np.argmin(t - [x.observation_time for x in epoch_index])
    # selected_epoch = epoch_index[epoch_i]

    # TODO: manual water calculation: show aperture vs r for oh/dust filters and allow picking windows to average over for counts to use for continuum subtraction
    # Then either take a redness --> Q or use an expectation value for Q based on redness distribution
    # Since this has counts we can calculate mag/flux etc. manually as well

    # TODO: blue spot detection, Q expectation value
    # TODO: aperture photometry on asymmetric leftover after profile subtraction

    # TODO: latex table generation from epoch index
    # ---------

    # regular lightcurve
    all_water_production_lightcurves(scp=scp, epoch_index=epoch_index)

    # bayesian lightcurve
    all_bayesian_water_production_lightcurves(scp=scp, epoch_index=epoch_index)

    # blue spot lightcurve
    all_blue_spot_lightcurves(scp=scp, epoch_index=epoch_index)

    # blue spot bayesian lightcurve
    all_bayesian_blue_spot_lightcurves(scp=scp, epoch_index=epoch_index)

    # TODO: active areas


if __name__ == "__main__":
    sys.exit(main())
