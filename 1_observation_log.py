#!/usr/bin/env python3

import os
import pathlib
import sys
from astropy.wcs.wcs import FITSFixedWarning
import yaml
import itertools
import warnings
import pandas as pd
import logging as log

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from typing import List, Optional
from argparse import ArgumentParser

from swift_types import (
    SwiftObservationID,
    SwiftUVOTImageType,
    SwiftFilterType,
    SwiftData,
    SwiftObservationLog,
)

__version__ = "0.0.1"


def read_yaml_from_file(filepath: pathlib.Path) -> Optional[dict]:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml


def observation_log(
    swift_data: SwiftData, obsids: List[SwiftObservationID], horizon_id: str
) -> Optional[SwiftObservationLog]:
    """
    Takes a list of observation ids and looks through all images that:
        - are from uvot
        - are sky_units (sk)
        - are in any filter
    and returns an observation log in the form of a pandas dataframe
    """

    image_type = SwiftUVOTImageType.sky_units
    all_filters = SwiftFilterType.all_filters()

    fits_header_entries_to_read = [
        "OBS_ID",
        "DATE-OBS",
        "DATE-END",
        "FILTER",
        "PA_PNT",
        "RA_OBJ",
        "DEC_OBJ",
        "EXPOSURE",
    ]
    obs_log = pd.DataFrame(columns=fits_header_entries_to_read)

    # list of every file we need to include in observation log?
    image_path_list = []
    # list of extensions that the
    extension_list = []
    # list of world coordinate systems of each image
    wcs_list = []
    # keep a list of the filenames that the extensions come from
    processed_filname_list = []

    for k, obsid in enumerate(obsids):
        print(f"Processing images for observation ID {obsid} ({k}/{len(obsids)}) ...")
        # get list of image file names from all filters that match the selected image type (sk, ex, ..)
        image_path_list = [
            swift_data.get_swift_uvot_image_paths(
                obsid=obsid, filter_type=filter_type, image_type=image_type
            )
            for filter_type in all_filters
        ]
        # filter out the ones that were not found
        image_path_list = list(filter(lambda x: x is not None, image_path_list))
        # flatten this list into 1d
        image_path_list = list(itertools.chain.from_iterable(image_path_list))

        # loop through every fits file found
        for image_path in image_path_list:
            # loop through every ImageHDU in the file
            with fits.open(image_path) as hdul:
                # skip the first extension, which should be informational
                for ext_id in range(1, len(hdul)):
                    # check if this extension is an image
                    if not isinstance(hdul[ext_id], fits.ImageHDU):
                        log.info(
                            "Skipping extension %s of fits file %s: not an ImageHDU",
                            ext_id,
                            image_path,
                        )
                        continue

                    header = hdul[ext_id].header  # type: ignore

                    # add a row to the dataframe from the header info
                    obs_log.loc[len(obs_log.index)] = [
                        header[i] for i in fits_header_entries_to_read
                    ]

                    wcs_list.append(WCS(header))
                    extension_list.append(ext_id)
                    processed_filname_list.append(image_path.name)

    # rename some columns
    obs_log = obs_log.rename(columns={"DATE-END": "DATE_END", "DATE-OBS": "DATE_OBS"})

    # convert the date columns from string to Time type
    obs_log["DATE_OBS"] = obs_log["DATE_OBS"].apply(lambda t: Time(t))
    obs_log["DATE_END"] = obs_log["DATE_END"].apply(lambda t: Time(t))

    # add middle of observation time
    dts = (obs_log["DATE_END"] - obs_log["DATE_OBS"]) / 2
    obs_log["MID_TIME"] = obs_log["DATE_OBS"] + dts

    # track which extension in the fits files the images are
    obs_log["EXTENSION"] = extension_list

    # the filename the extension was pulled from
    obs_log["FITS_FILENAME"] = processed_filname_list

    ephemeris_info = {
        "r": "HELIO",
        "r_rate": "HELIO_V",
        "delta": "OBS_DIS",
        "alpha": "PHASE",
        "RA": "RA",
        "DEC": "DEC",
    }
    # make dataframe with columns of the ephemeris_info values
    horizon_dataframe = pd.DataFrame(columns=list(ephemeris_info.values()))

    # for each row, query Horizons for our object at 'mid_time' and fill the dataframe with response info
    for k, mid_time in enumerate(obs_log["MID_TIME"]):
        print(
            f"Querying JPL Horizons for {obs_log['OBS_ID'][k]} extension {obs_log['EXTENSION'][k]} ({k+1}/{obs_log.shape[0]})"
        )
        horizon_response = Horizons(
            id=horizon_id, location="@swift", epochs=mid_time.jd, id_type="smallbody"
        )
        eph = horizon_response.ephemerides()  # type: ignore
        # append this row of information to our horizon dataframe
        horizon_dataframe.loc[len(horizon_dataframe.index)] = [
            eph[x][0] for x in ephemeris_info.keys()
        ]

    obs_log = pd.concat([obs_log, horizon_dataframe], axis=1)

    x_list = []
    y_list = []
    # use the positions found from Horizons to find the pixel center of the comet based on its image WCS
    for i, (ra, dec) in enumerate(zip(obs_log["RA"], obs_log["DEC"])):
        x, y = wcs_list[i].wcs_world2pix(ra, dec, 1)
        x_list.append(x)
        y_list.append(y)

    obs_log["PX"] = x_list
    obs_log["PY"] = y_list

    return obs_log


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
        "--output",
        "-o",
        default="swift_observation_log.csv",
        help="Filename of observation log output",
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
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    swift_config = read_yaml_from_file(pathlib.Path(args.config))
    if swift_config is None:
        print("Error reading config file {args.config}, exiting.")
        return 1

    horizon_id = swift_config["jpl_horizons_id"]

    sdd = SwiftData(
        data_path=pathlib.Path(swift_config["swift_data_dir"]).expanduser().resolve()
    )

    # # TODO: remove this small dataset after testing
    # small_dataset = False
    #
    # if small_dataset:
    #     obsid_strings = ["00020405001", "00020405002", "00020405003", "00034318005"]
    #     obsids = list(map(lambda x: SwiftObservationID(x), obsid_strings))
    #     df = observation_log(swift_data=sdd, obsids=obsids, horizon_id=horizon_id)
    # else:
    #     df = observation_log(
    #         swift_data=sdd, obsids=sdd.get_all_observation_ids(), horizon_id=horizon_id
    #     )

    df = observation_log(
        swift_data=sdd, obsids=sdd.get_all_observation_ids(), horizon_id=horizon_id
    )

    if df is None:
        print(
            "Could not construct the observation log in memory, exiting without writing output."
        )
        return 1

    # write out our dataframe
    df.to_csv(args.output)


if __name__ == "__main__":
    sys.exit(main())
