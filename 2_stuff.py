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

# from typing import Optional, TypeAlias, List
from argparse import ArgumentParser

from swift_types import (
    SwiftObservationID,
    SwiftUVOTImageType,
    SwiftFilterType,
    SwiftData,
    # SwiftObservationLog,
    swift_observation_id_from_int,
)


__version__ = "0.0.1"


# # TODO: fix type for ImageHDU
# def center_image_on_coords(image: fits.ImageHDU, old_coords_to_center, size_of_new_image):
#     """
#     size is the (rows, columns) size of the positive quandrant of the new image
#     """
#
#     pass


def set_coord(image_array, target_index, size):
    """To shift a target on an image
    into the center of a new image;

    The size of the new image can be given
    but have to ensure the whole original
    image is included in the new one.

    Inputs: array of an original image, 2D array
            original coordinate values of the target, array shape in [r, c]
            output size, tuple of 2 elements
    Outputs: array of the shifted image in the new coordinate, 2D array
    """
    # interpret the size and create new image
    try:
        half_row, half_col = size
    except:
        print("Check the given image size!")
    new_coord = np.zeros((2 * half_row - 1, 2 * half_col - 1))

    # shift the image, [target] -> [center]
    def shift_r(r):
        return int(r + (half_row - 1 - target_index[0]))

    def shift_c(c):
        return int(c + (half_col - 1 - target_index[1]))

    for r in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            new_coord[shift_r(r), shift_c(c)] = image_array[r, c]
    # reture new image
    return new_coord


def stack_image(obs_log_name, filt, size, output_name):
    """sum obs images according to 'FILTER'

    Inputs:
    obs_log_name: the name of an obs log in docs/
    filt: 'uvv' or 'uw1' or 'uw2'
    size: a tuple
    output_name: string, to be saved in docs/

    Outputs:
    1) a txt data file saved in docs/
    2) a fits file saved in docs/
    """
    # load obs_log in DataFrame according to filt
    obs_log_path = get_path("../docs/" + obs_log_name)
    img_set = pd.read_csv(obs_log_path, sep=" ", index_col=["FILTER"])
    img_set = img_set[
        ["OBS_ID", "EXTENSION", "PX", "PY", "PA", "EXP_TIME", "END", "START"]
    ]
    if filt == "uvv":
        img_set = img_set.loc["V"]
    elif filt == "uw1":
        img_set = img_set.loc["UVW1"]
    elif filt == "uw2":
        img_set = img_set.loc["UVW2"]
    # ---transfer OBS_ID from int to string---
    img_set["OBS_ID"] = img_set["OBS_ID"].astype(str)
    img_set["OBS_ID"] = "000" + img_set["OBS_ID"]
    # create a blank canvas in new coordinate
    stacked_img = np.zeros((2 * size[0] - 1, 2 * size[1] - 1))
    # loop among the data set, for every image, shift it to center the target, rotate and add to the blank canvas
    exp = 0
    for i in range(len(img_set)):
        # ---get data from .img.gz---
        if img_set.index.name == "FILTER":
            img_now = img_set.iloc[i]
        else:
            img_now = img_set
        img_path = get_path(img_now["OBS_ID"], filt, to_file=True)
        img_hdu = fits.open(img_path)[img_now["EXTENSION"]]
        img_data = img_hdu.data.T  # .T! or else hdul PXY != DS9 PXY
        # ---shift the image to center the target---
        new_img = set_coord(
            img_data, np.array([img_now["PX"] - 1, img_now["PY"] - 1]), size
        )
        # ---rotate the image according to PA to
        # ---eliminate changes of pointing
        # ---this rotating step may be skipped---
        # new_img = rotate(new_img,
        #                 angle=img_now['PA'],
        #                 reshape=False,
        #                 order=1)
        # ---sum modified images to the blank canvas---
        stacked_img = stacked_img + new_img
        exp += img_now["EXP_TIME"]
    # get the summed results and save in fits file
    output_path = get_path("../docs/" + output_name + "_" + filt + ".fits")
    hdu = fits.PrimaryHDU(stacked_img)
    if img_set.index.name == "FILTER":
        dt = Time(img_set.iloc[-1]["END"]) - Time(img_set.iloc[0]["START"])
        mid_t = Time(img_set.iloc[0]["START"]) + 1 / 2 * dt
    else:
        dt = Time(img_set["END"]) - Time(img_set["START"])
        mid_t = Time(img_set["START"]) + 1 / 2 * dt
    hdr = hdu.header
    hdr["TELESCOP"] = img_hdu.header["TELESCOP"]
    hdr["INSTRUME"] = img_hdu.header["INSTRUME"]
    hdr["FILTER"] = img_hdu.header["FILTER"]
    hdr["COMET"] = obs_log_name.split("_")[0] + " " + obs_log_name.split("_")[-1][:-4]
    hdr["PLATESCL"] = ("1", "arcsec/pixel")
    hdr["XPOS"] = f"{size[0]}"
    hdr["YPOS"] = f"{size[1]}"
    hdr["EXPTIME"] = (f"{exp}", "[seconds]")
    hdr["MID_TIME"] = f"{mid_t}"
    hdu.writeto(output_path)


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
        "swiftdatadir", nargs=1, help="top-level directory of Swift data"
    )  # the nargs=? specifies 0 or 1 arguments: it is optional
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
    sdd = SwiftData(data_path=pathlib.Path(args.swiftdatadir[0]).expanduser())

    df = pd.read_csv(args.observation_log_file[0])

    # obsid_strings = ["00034318005"]
    # obsid_strings = ["00020405001", "00020405002", "00020405003", "00034318005"]
    obsid_strings = ["00020405003"]
    obsids = list(map(lambda x: SwiftObservationID(x), obsid_strings))

    for _, row in df.iterrows():
        obsid = swift_observation_id_from_int(row["OBS_ID"])  # type: ignore
        if obsid is None:
            continue
        if obsid != obsids[0]:
            continue

        for filt in [SwiftFilterType.uw1]:
            for image_type in [SwiftUVOTImageType.sky_units]:
                img_file_list = sdd.get_swift_uvot_image_paths(
                    obsid=obsid, filter_type=filt, image_type=image_type
                )
                if img_file_list is None:
                    continue
                print(img_file_list)
                for img_file in img_file_list:
                    print(row["PX"], row["PY"], row["EXTENSION"])
                    image_data = fits.getdata(img_file, ext=row["EXTENSION"])
                    # numpy uses (row, col) indexing - so (y, x)
                    new_image = set_coord(
                        image_data,
                        np.array([row["PY"] - 1, row["PX"] - 1]),
                        (1500, 1500),
                    )
                    show_fits_scaled(image_data, new_image)


if __name__ == "__main__":
    sys.exit(main())
