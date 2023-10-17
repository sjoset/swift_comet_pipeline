#!/usr/bin/env python3

from functools import reduce
import os
import pathlib
import sys
import logging as log
from argparse import ArgumentParser
from astropy.time import Time

import numpy as np

# import matplotlib.pyplot as plt
# from numpy import unique

# from astropy.visualization import ZScaleInterval

from configs import read_swift_project_config
from epochs import Epoch
from observation_log import get_image_path_from_obs_log_row
from pipeline_files import PipelineFiles

from swift_data import SwiftData
from swift_filter import SwiftFilter, filter_to_file_string
from tui import epoch_menu


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
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


# def show_fits_subtracted(uw1, uvv, beta):
#     # adjust for the different count rates between filters
#     beta *= 6.0
#
#     dust_map = -(uw1 - beta * uvv)
#
#     # dust_scaled = np.log10(np.clip(dust_map, 0, None) + eps)
#     dust_scaled = np.log10(dust_map)
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#
#     zscale = ZScaleInterval()
#     vmin, vmax = zscale.get_limits(dust_scaled)
#
#     im1 = ax1.imshow(dust_scaled, vmin=vmin, vmax=vmax)
#     # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
#     fig.colorbar(im1)
#
#     image_center_row = int(np.floor(uw1.shape[0] / 2))
#     image_center_col = int(np.floor(uw1.shape[1] / 2))
#     ax1.axvline(image_center_col, color="b", alpha=0.2)
#     ax1.axhline(image_center_row, color="b", alpha=0.2)
#
#     # hdu = fits.PrimaryHDU(dust_subtracted)
#     # hdu.writeto("subtracted.fits", overwrite=True)
#     plt.show()


# def show_centers(img, cs: List[PixelCoord]):
#     # img_scaled = np.log10(img)
#     img_scaled = img
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#
#     pix_center = get_uvot_image_center(img)
#     ax1.add_patch(
#         plt.Circle(
#             (pix_center.x, pix_center.y),
#             radius=30,
#             fill=False,
#         )
#     )
#
#     zscale = ZScaleInterval()
#     vmin, vmax = zscale.get_limits(img_scaled)
#
#     im1 = ax1.imshow(img_scaled, vmin=vmin, vmax=vmax)
#     fig.colorbar(im1)
#
#     for c in cs:
#         line_color = next(ax1._get_lines.prop_cycler)["color"]
#         ax1.axvline(c.x, alpha=0.7, color=line_color)
#         ax1.axhline(c.y, alpha=0.9, color=line_color)
#
#     plt.show()


def full_epoch_summary(swift_data: SwiftData, pipeline_files: PipelineFiles) -> None:
    epoch_product = epoch_menu(pipeline_files)
    if epoch_product is None:
        print("Could not select epoch!")
        return
    epoch_product.load_product()
    epoch = epoch_product.data_product

    for filter_type in [SwiftFilter.uvv, SwiftFilter.uw1]:
        epoch_mask = epoch.FILTER == filter_type
        filtered_epoch = epoch[epoch_mask]
        print(f"Observations in filter {filter_to_file_string(filter_type)}:")

        print(filtered_epoch)
        imglist = set(
            [
                str(
                    get_image_path_from_obs_log_row(
                        swift_data=swift_data, obs_log_row=row
                    )
                )
                for _, row in filtered_epoch.iterrows()
            ]
        )
        print("Fits images used:")
        for img in imglist:
            print(img)

        print(
            f"Total exposure time in filter {filter_to_file_string(filter_type)}: {np.sum(filtered_epoch.EXPOSURE)}"
        )
        print("")


def latex_table_summary(epoch: Epoch) -> None:
    print("")
    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        epoch_mask = epoch.FILTER == filter_type
        filtered_epoch = epoch[epoch_mask]

        obs_times = [Time(x) for x in filtered_epoch.MID_TIME]  # type: ignore
        obs_dates = [x.to_datetime().date() for x in obs_times]  # type: ignore

        unique_days = np.unique(obs_dates)
        unique_days_str = reduce(lambda x, y: str(x) + ", " + str(y), unique_days)

        num_images = len(filtered_epoch)
        exposure_time = np.sum(filtered_epoch.EXPOSURE)
        rh = np.mean(filtered_epoch.HELIO)
        delta = np.mean(filtered_epoch.OBS_DIS)

        # print("Obs Date & Filter & Images & Exposure Time & R_h & delta")
        print(
            f" & {unique_days_str} & {filter_to_file_string(filter_type)} & {num_images} & {exposure_time:4.0f} & {rh:3.2f} & {delta:3.2f} \\\\"
        )


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    full_epoch_summary(swift_data=swift_data, pipeline_files=pipeline_files)

    # if pipeline_files.epoch_products is None:
    #     print("No epoch files found! Exiting.")
    #     return 0
    #
    # for epoch_product in pipeline_files.epoch_products:
    #     if epoch_product is None:
    #         print(f"Skipping epoch {epoch_product.product_path}!")
    #         continue
    #     epoch_product.load_product()
    #     epoch = epoch_product.data_product
    #     latex_table_summary(epoch=epoch)

    # TODO: generate Konrad plots also


if __name__ == "__main__":
    sys.exit(main())
