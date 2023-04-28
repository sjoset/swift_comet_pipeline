#!/usr/bin/env python3

import os
import pathlib
import sys
import pandas as pd
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from astropy.visualization import (
    ZScaleInterval,
)
from astropy.io import fits
from astropy.time import Time
from typing import Tuple, Optional
from argparse import ArgumentParser

# from itertools import product

from swift_types import (
    SwiftObservationID,
    SwiftObservationLog,
    SwiftFilter,
    SwiftImage,
    PixelCoord,
    swift_observation_id_from_int,
)
from read_swift_config import read_swift_config
from swift_data import SwiftData
from swift_observation_log import (
    get_observation_log_rows_that_match,
)
from sushi import set_coord


__version__ = "0.0.1"


def get_image_dimensions_to_center_comet(
    source_image: SwiftImage, source_coords_to_center: PixelCoord
) -> Tuple[float, float]:
    """
    If we want to re-center the source_image on source_coords_to_center, we need to figure out the dimensions the new image
    needs to be to fit the old picture after we scoot it over to its new position
    Returns tuple of (rows, columns) the new image would have to be
    """
    num_rows, num_columns = source_image.shape
    center_row = num_rows / 2.0
    center_col = num_columns / 2.0

    # distance the image needs to move
    d_rows = center_row - source_coords_to_center.y
    d_cols = center_col - source_coords_to_center.x

    # the total padding is twice the distance it needs to move -
    # extra space for it to move into, and the empty space it would leave behind
    row_padding = 2 * d_rows
    col_padding = 2 * d_cols

    retval = (
        np.ceil(source_image.shape[0] + np.abs(row_padding)),
        np.ceil(source_image.shape[1] + np.abs(col_padding)),
    )

    return retval


def determine_image_resize_dimensions_by_obsid(
    swift_data: SwiftData,
    obs_log: SwiftObservationLog,
    obsid: SwiftObservationID,
    filter_type: SwiftFilter,
) -> Optional[Tuple[int, int]]:
    image_dir = swift_data.get_uvot_image_directory(obsid)

    # dataframe holding the observation log entries for this obsid and this filter
    matching_observations = get_observation_log_rows_that_match(
        obs_log, obsid, filter_type
    )

    if len(matching_observations) == 0:
        return None

    # how far in pixels each comet image needs to shift
    image_dimensions = []

    # loop through each dataframe row
    for _, row in matching_observations.iterrows():
        # open file
        img_file = image_dir / pathlib.Path(str(row["FITS_FILENAME"]))
        image_data = fits.getdata(img_file, ext=row["EXTENSION"])

        # keep a list of the image sizes
        dimensions = get_image_dimensions_to_center_comet(
            image_data, PixelCoord(x=row["PX"], y=row["PY"])  # type: ignore
        )
        image_dimensions.append(dimensions)

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(image_dimensions, key=lambda k: k[0], reverse=True)[0][0]
    max_num_cols = sorted(image_dimensions, key=lambda k: k[1], reverse=True)[0][1]

    # how many extra pixels we need
    return (max_num_rows, max_num_cols)


def center_image_on_coords(
    source_image: SwiftImage,
    source_coords_to_center: PixelCoord,
    size_of_new_image: Tuple[int, int],
) -> SwiftImage:
    """
    size is the (rows, columns) size of the positive quandrant of the new image
    """

    center_x, center_y = np.round(source_coords_to_center.x), np.round(
        source_coords_to_center.y
    )
    new_r, new_c = map(lambda x: int(x), np.ceil(size_of_new_image))

    # enforce that we have an odd number of pixels so that the comet can be moved to the center row, center column
    if new_r % 2 == 0:
        half_r = new_r / 2
        new_r = new_r + 1
    else:
        half_r = (new_r - 1) / 2
    if new_c % 2 == 0:
        half_c = new_c / 2
        new_c = new_c + 1
    else:
        half_c = (new_c - 1) / 2

    # create empty array to hold the new, centered image
    centered_image = np.zeros((new_r, new_c))

    def shift_row(r):
        return int(r + half_r + 1 - center_y)

    def shift_column(c):
        return int(c + half_c + 1 - center_x)

    for r in range(source_image.shape[0]):
        for c in range(source_image.shape[1]):
            centered_image[shift_row(r), shift_column(c)] = source_image[r, c]

    print(f">> Mid x: {half_c} ||| Mid y: {half_r}")
    print(shift_row(center_y), shift_column(center_x), half_r, half_c)

    return centered_image


# TODO: move this over
def coincidence_loss_correction(image_data: SwiftImage) -> None:
    return


# def stack(obs_log_name, filt, size, output_name):
#     """sum obs images according to 'FILTER'
#
#     Inputs:
#     obs_log_name: the name of an obs log in docs/
#     filt: 'uvv' or 'uw1' or 'uw2'
#     size: a tuple
#     output_name: string, to be saved in docs/
#
#     Outputs:
#     1) a txt data file saved in docs/
#     2) a fits file saved in docs/
#     """
#     # load obs_log in DataFrame according to filt
#     obs_log_path = get_path("../docs/" + obs_log_name)
#     img_set = pd.read_csv(obs_log_path, sep=" ", index_col=["FILTER"])
#     img_set = img_set[
#         ["OBS_ID", "EXTENSION", "PX", "PY", "PA", "EXP_TIME", "END", "START"]
#     ]
#     if filt == "uvv":
#         img_set = img_set.loc["V"]
#     elif filt == "uw1":
#         img_set = img_set.loc["UVW1"]
#     elif filt == "uw2":
#         img_set = img_set.loc["UVW2"]
#     # ---transfer OBS_ID from int to string---
#     img_set["OBS_ID"] = img_set["OBS_ID"].astype(str)
#     img_set["OBS_ID"] = "000" + img_set["OBS_ID"]
#     # create a blank canvas in new coordinate
#     stacked_img = np.zeros((2 * size[0] - 1, 2 * size[1] - 1))
#     # loop among the data set, for every image, shift it to center the target, rotate and add to the blank canvas
#     exp = 0
#     for i in range(len(img_set)):
#         # ---get data from .img.gz---
#         if img_set.index.name == "FILTER":
#             img_now = img_set.iloc[i]
#         else:
#             img_now = img_set
#         img_path = get_path(img_now["OBS_ID"], filt, to_file=True)
#         img_hdu = fits.open(img_path)[img_now["EXTENSION"]]
#         img_data = img_hdu.data.T  # .T! or else hdul PXY != DS9 PXY
#         # ---shift the image to center the target---
#         new_img = set_coord(
#             img_data, np.array([img_now["PX"] - 1, img_now["PY"] - 1]), size
#         )
#         # ---rotate the image according to PA to
#         # ---eliminate changes of pointing
#         # ---this rotating step may be skipped---
#         # new_img = rotate(new_img,
#         #                 angle=img_now['PA'],
#         #                 reshape=False,
#         #                 order=1)
#         # ---sum modified images to the blank canvas---
#         stacked_img = stacked_img + new_img
#         exp += img_now["EXP_TIME"]
#     # get the summed results and save in fits file
#     output_path = get_path("../docs/" + output_name + "_" + filt + ".fits")
#     hdu = fits.PrimaryHDU(stacked_img)
#     if img_set.index.name == "FILTER":
#         dt = Time(img_set.iloc[-1]["END"]) - Time(img_set.iloc[0]["START"])
#         mid_t = Time(img_set.iloc[0]["START"]) + 1 / 2 * dt
#     else:
#         dt = Time(img_set["END"]) - Time(img_set["START"])
#         mid_t = Time(img_set["START"]) + 1 / 2 * dt
#     hdr = hdu.header
#     hdr["TELESCOP"] = img_hdu.header["TELESCOP"]
#     hdr["INSTRUME"] = img_hdu.header["INSTRUME"]
#     hdr["FILTER"] = img_hdu.header["FILTER"]
#     hdr["COMET"] = obs_log_name.split("_")[0] + " " + obs_log_name.split("_")[-1][:-4]
#     hdr["PLATESCL"] = ("1", "arcsec/pixel")
#     hdr["XPOS"] = f"{size[0]}"
#     hdr["YPOS"] = f"{size[1]}"
#     hdr["EXPTIME"] = (f"{exp}", "[seconds]")
#     hdr["MID_TIME"] = f"{mid_t}"
#     hdu.writeto(output_path)


def show_fits_scaled(image_data, new_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)

    im1 = ax1.imshow(image_data, vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(new_data, vmin=vmin, vmax=vmax)
    fig.colorbar(im1)
    fig.colorbar(im2)

    plt.show()


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
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
        "observation_log_file", nargs=1, help="Filename of observation log input"
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


def main():
    args = process_args()

    swift_config = read_swift_config(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    sdd = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    obs_log = pd.read_csv(args.observation_log_file[0])

    # obsid_strings = ["00034318005"]
    obsid_strings = ["00020405001", "00020405002", "00020405003", "00034318005"]
    # obsid_strings = ["00020405003"]
    # obsid_strings = [
    #     "00034316001",
    #     "00034316002",
    #     "00034318001",
    #     "00034318002",
    #     "00034318003",
    # ]
    obsids = list(map(lambda x: SwiftObservationID(x), obsid_strings))

    # loop through every row of the observation log
    for _, row in obs_log.iterrows():
        obsid = swift_observation_id_from_int(row["OBS_ID"])  # type: ignore
        if obsid is None:
            print(f"Failed getting {obsid=} from the observing log, skipping entry.")
            continue
        if obsid not in obsids:
            continue

        for filter_type in [SwiftFilter.uvv]:
            # Get list of image files for this filter and image type
            img_file_list = sdd.get_swift_uvot_image_paths(
                obsid=obsid, filter_type=filter_type
            )
            if img_file_list is None:
                # print(f"No files found for {obsid=} and {filter_type=}")
                continue

            new_image_size = determine_image_resize_dimensions_by_obsid(
                swift_data=sdd,
                obs_log=obs_log,
                obsid=obsid,
                filter_type=filter_type,
            )

            for img_file in img_file_list:
                print(
                    f"Processing {img_file.name}, extension {row['EXTENSION']}: Comet center at {row['PX']}, {row['PY']}"
                )

                image_data = fits.getdata(img_file, ext=row["EXTENSION"])

                # numpy uses (row, col) indexing - so (y, x)
                new_image2 = set_coord(
                    image_data,
                    np.array([row["PY"] - 1, row["PX"] - 1]),
                    (1500, 1500),
                )

                new_image = center_image_on_coords(
                    image_data,
                    PixelCoord(x=row["PX"], y=row["PY"]),
                    size_of_new_image=new_image_size,
                )
                # show_fits_scaled(image_data, new_image)
                show_fits_scaled(new_image, new_image2)
                print("-------")


if __name__ == "__main__":
    sys.exit(main())
