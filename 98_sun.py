#!/usr/bin/env python3

import os
import glob
import pathlib
import warnings
import sys
import logging as log
import numpy as np

import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from argparse import ArgumentParser

from configs import read_swift_project_config
from pipeline_files import PipelineFiles
from swift_data import SwiftData

from epochs import read_epoch, write_epoch
from manual_veto import manual_veto
from swift_filter import SwiftFilter
from user_input import get_yes_no, get_selection

from astropy.coordinates import get_body_barycentric, get_sun
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.wcs import FITSFixedWarning


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config", nargs=1, help="Filename of project config"
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


def show_fits(img, sun_x, sun_y, comet_x, comet_y):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img)

    im1 = ax1.imshow(img, vmin=vmin, vmax=vmax)
    # im2 = ax2.imshow(image_median, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)

    image_center_row = int(np.floor(img.shape[0] / 2))
    image_center_col = int(np.floor(img.shape[1] / 2))
    # ax1.plot([sun_x, image_center_col, -sun_x], [sun_y, image_center_row, -sun_y])
    ax1.plot([sun_x, comet_x], [sun_y, comet_y])
    ax1.set_xlim(0, img.shape[1])
    ax1.set_ylim(0, img.shape[0])
    # ax1.axvline(image_center_col, color="b", alpha=0.2)
    # ax1.axhline(image_center_row, color="b", alpha=0.2)

    # hdu = fits.PrimaryHDU(dust_subtracted)
    # hdu.writeto("subtracted.fits", overwrite=True)
    plt.show()


def epoch_menu(epoch_dir: pathlib.Path) -> pathlib.Path:
    glob_pattern = str(epoch_dir / pathlib.Path("*.parquet"))

    epoch_filename_list = sorted(glob.glob(glob_pattern))

    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])

    return epoch_path


def main():
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_path = epoch_menu(pipeline_files.epoch_dir_path)
    epoch_pre_veto = read_epoch(epoch_path)

    filter_mask = epoch_pre_veto["FILTER"] == SwiftFilter.uvv
    epoch_pre_veto = epoch_pre_veto[filter_mask]

    for _, row in epoch_pre_veto.iterrows():
        img = swift_data.get_uvot_image(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        wcs = swift_data.get_uvot_image_wcs(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        print(wcs)

        print(f"t = {Time(row.MID_TIME)}")
        sun = get_sun(Time(row.MID_TIME))
        print(f"{sun=}")

        # print(wcs.wcs_world2pix(sun))
        sun_x, sun_y = skycoord_to_pixel(sun, wcs)
        print(sun_x, sun_y, np.degrees(np.arctan2(sun_y, sun_x)))

        show_fits(img, sun_x, sun_y, row.PX, row.PY)

    # image_path = swift_data.get_uvot_image_directory(obsid=row.OBS_ID) / pathlib.Path(
    #     row["FITS_FILENAME"]
    # )
    #
    # with fits.open(image_path) as hdul:
    #     header = hdul[ext_id].header  # type: ignore
    #     wcs_list.append(WCS(header))

    # epoch_post_veto = manual_veto(
    #     swift_data=swift_data, epoch=epoch_pre_veto, epoch_title=epoch_path.stem
    # )
    #
    # print("Save epoch?")
    # save_epoch = get_yes_no()
    # if not save_epoch:
    #     return
    #
    # write_epoch(epoch=epoch_post_veto, epoch_path=epoch_path)


if __name__ == "__main__":
    sys.exit(main())
