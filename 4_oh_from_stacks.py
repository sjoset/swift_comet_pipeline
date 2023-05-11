#!/usr/bin/env python3

import os
import pathlib
import sys
import numpy as np
import logging as log

from astropy.time import Time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)
from typing import Tuple

from read_swift_config import read_swift_config
from swift_types import SwiftFilter, SwiftStackingMethod, SwiftUVOTImage
from reddening_correction import DustReddeningPercent
from stack_info import stacked_images_from_stackinfo
from fluorescence_OH import flux_OH_to_num_OH, read_gfactor_1au
from flux_OH import OH_flux_from_count_rate, OH_flux_from_count_rate_fixed_beta
from aperture_photometry import do_aperture_photometry
from vectorial_model import num_OH_to_Q

__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="YAML configuration file to use"
    )
    parser.add_argument(
        "stackinfo",
        default=None,
        nargs="?",
        help="JSON file containing stacking information",
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


def show_fits_scaled(image_sum, image_median):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_sum)

    im1 = ax1.imshow(image_sum, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    plt.show()


def pad_to_match_sizes(
    uw1: SwiftUVOTImage, uvv: SwiftUVOTImage
) -> Tuple[SwiftUVOTImage, SwiftUVOTImage]:
    # pads the edges of the smaller image so that the two images share dimensions, allowing the dust-subtraction uw1 - beta * uvv to be visualized
    cols_to_add = round((uw1.shape[1] - uvv.shape[1]) / 2)
    rows_to_add = round((uw1.shape[0] - uvv.shape[0]) / 2)

    if cols_to_add > 0:
        # uw1 is larger, pad uvv to be larger
        uvv = np.pad(
            uvv,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        cols_to_add = np.abs(cols_to_add)
        uw1 = np.pad(
            uw1,
            ((0, 0), (cols_to_add, cols_to_add)),
            mode="constant",
            constant_values=0.0,
        )

    if rows_to_add > 0:
        # uw1 is larger, pad uvv to be larger
        uvv = np.pad(
            uvv,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        rows_to_add = np.abs(rows_to_add)
        uw1 = np.pad(
            uw1,
            ((rows_to_add, rows_to_add), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    return (uw1, uvv)


def show_fits_subtracted(uw1_stack, uvv_stack, beta):
    uw1 = uw1_stack.stacked_image
    uvv = uvv_stack.stacked_image

    uw1, uvv = pad_to_match_sizes(uw1, uvv)
    dust_subtracted = uw1 - beta * uvv

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(dust_subtracted)

    im1 = ax1.imshow(dust_subtracted, vmin=vmin, vmax=vmax)
    # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    plt.show()


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    if args.stackinfo is None:
        print("Stack info file not specified, trying default filename stack.json ...")
        stackinfo_path = pathlib.Path("stack.json")
    else:
        stackinfo_path = args.stackinfo

    print(f"Loading stacked image information from {stackinfo_path}")
    stacked_images = stacked_images_from_stackinfo(stackinfo_path=stackinfo_path)

    # collect aperture results for uw1 and uvv filters from the stacked images
    aperture_results = {}
    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        summed_image = stacked_images[(filter_type, SwiftStackingMethod.summation)]
        median_image = stacked_images[(filter_type, SwiftStackingMethod.median)]
        # show_fits_scaled(summed_image.stacked_image, median_image.stacked_image)
        aperture_results[filter_type] = do_aperture_photometry(
            summed_image, median_image
        )

    dust_redness = DustReddeningPercent(10)
    # calculate OH flux based on the aperture results in uw1 and uvv filters
    print(f"\n\nCalculating OH flux with dust redness {dust_redness.reddening}% ...")
    flux_OH_1 = OH_flux_from_count_rate(
        solar_spectrum_path=swift_config["solar_spectrum_path"],
        # solar_spectrum_time=Time("2457422", format="jd"),
        # solar_spectrum_time=Time("2016-02-01"),
        solar_spectrum_time=Time(
            stacked_images[
                (SwiftFilter.uw1, SwiftStackingMethod.summation)
            ].observation_mid_time,
            format="fits",
        ),
        effective_area_uw1_path=swift_config["effective_area_uw1_path"],
        effective_area_uvv_path=swift_config["effective_area_uvv_path"],
        result_uw1=aperture_results[SwiftFilter.uw1],
        result_uvv=aperture_results[SwiftFilter.uvv],
        dust_redness=dust_redness,
    )
    flux_OH_2 = OH_flux_from_count_rate_fixed_beta(
        effective_area_uw1_path=swift_config["effective_area_uw1_path"],
        effective_area_uvv_path=swift_config["effective_area_uvv_path"],
        result_uw1=aperture_results[SwiftFilter.uw1],
        result_uvv=aperture_results[SwiftFilter.uvv],
        dust_redness=dust_redness,
    )

    fluorescence_data = read_gfactor_1au(swift_config["oh_fluorescence_path"])
    img = stacked_images[(SwiftFilter.uw1, SwiftStackingMethod.summation)]
    num_OH_1 = flux_OH_to_num_OH(
        flux_OH=flux_OH_1,
        helio_r_au=img.helio_r_au,
        helio_v_kms=img.helio_v_kms,
        delta_au=img.delta_au,
        fluorescence_data=fluorescence_data,
    )
    num_OH_2 = flux_OH_to_num_OH(
        flux_OH=flux_OH_2,
        helio_r_au=img.helio_r_au,
        helio_v_kms=img.helio_v_kms,
        delta_au=img.delta_au,
        fluorescence_data=fluorescence_data,
    )
    print("")
    print(f"Total number of OH, beta from solar spectrum method: {num_OH_1}")
    print(f"Total number of OH, fixed beta method: {num_OH_2}")

    Q1 = num_OH_to_Q(
        helio_r=img.helio_r_au,
        num_OH=num_OH_1,
        vectorial_model_path=swift_config["vectorial_model_path"],
    )
    Q2 = num_OH_to_Q(
        helio_r=img.helio_r_au,
        num_OH=num_OH_2,
        vectorial_model_path=swift_config["vectorial_model_path"],
    )
    print("")
    print(f"Heliocentric comet distance: {img.helio_r_au}")
    print(f"Q(H2O) from N(OH), first method: {Q1} mol/s at {img.observation_mid_time}")
    print(f"Q(H2O) from N(OH), second method: {Q2} mol/s at {img.observation_mid_time}")

    show_fits_subtracted(
        stacked_images[(SwiftFilter.uw1, SwiftStackingMethod.summation)],
        stacked_images[(SwiftFilter.uvv, SwiftStackingMethod.summation)],
        beta=0.101483,
    )
    show_fits_subtracted(
        stacked_images[(SwiftFilter.uw1, SwiftStackingMethod.median)],
        stacked_images[(SwiftFilter.uvv, SwiftStackingMethod.median)],
        beta=0.101483,
    )


if __name__ == "__main__":
    sys.exit(main())
