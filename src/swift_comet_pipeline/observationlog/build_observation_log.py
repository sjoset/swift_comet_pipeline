import itertools

import logging as log
import numpy as np
import pandas as pd
import astropy.units as u

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from tqdm import tqdm

from swift_comet_pipeline.comet.comet_center import invalid_user_center_value
from swift_comet_pipeline.observationlog.observation_log import SwiftObservationLog
from swift_comet_pipeline.swift.swift_data import (
    SwiftData,
    swift_observation_id_from_int,
    swift_orbit_id_from_obsid,
)
from swift_comet_pipeline.swift.swift_filter import (
    SwiftFilter,
    obs_string_to_filter,
)
from swift_comet_pipeline.swift.uvot_image import (
    datamode_from_fits_keyword_string,
    datamode_to_pixel_resolution,
)


def build_observation_log(
    swift_data: SwiftData,
    horizons_id: str,
) -> SwiftObservationLog | None:
    """
    Takes a swift data structure and looks through observation ids that have images that:
        - are from uvot
        - are sky_units (sk)
        - are in any filter
    and returns an observation log in the form of a pandas dataframe
    """

    all_filters = SwiftFilter.all_filters()
    obsids = swift_data.get_all_observation_ids()

    # TODO: the entry DATAMODE is 'IMAGE' - confirm this is data mode and not event mode

    fits_header_entries_to_read = [
        "OBS_ID",
        "DATE-OBS",
        "DATE-END",
        "FILTER",
        "PA_PNT",
        "RA_OBJ",
        "DEC_OBJ",
        "EXPOSURE",
        "DATAMODE",
    ]
    obs_log = pd.DataFrame(columns=fits_header_entries_to_read)  # type: ignore

    # list of every file we need to include in observation log?
    image_path_list = []
    # list of extensions that describe which sub-image inside a FITS file we are describing
    extension_list = []
    # list of world coordinate systems of each image
    wcs_list = []
    # keep a list of the filenames that the extensions come from
    processed_filname_list = []
    # swift pipeline versions that produced our downloaded data
    creator_list = []
    # image_shape_row_list = []
    # image_shape_col_list = []

    image_progress_bar = tqdm(obsids, unit="images")
    for k, obsid in enumerate(image_progress_bar):
        # get list of image file names from all filters that match the selected image type (sk, ex, ..)
        image_path_list = [
            swift_data.get_swift_uvot_image_paths(obsid=obsid, filter_type=filter_type)
            for filter_type in all_filters
        ]
        # filter out the ones that were not found
        image_path_list = list(filter(lambda x: x is not None, image_path_list))
        # flatten this list into 1d
        image_path_list = list(itertools.chain.from_iterable(image_path_list))  # type: ignore

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

                    # # pull the numpy shape out of the pixel array
                    # img_shape = hdul[ext_id].data.shape  # type: ignore
                    # image_shape_row_list.append(img_shape[0])
                    # image_shape_col_list.append(img_shape[1])

                    # add a row to the dataframe from the header info
                    obs_log.loc[len(obs_log.index)] = [
                        header[i] for i in fits_header_entries_to_read
                    ]

                    wcs_list.append(WCS(header))
                    extension_list.append(ext_id)
                    processed_filname_list.append(image_path.name)  # type: ignore

                    image_progress_bar.set_description(f"Image: {image_path.name}")  # type: ignore
                    creator_list.append(hdul[0].header["CREATOR"])  # type: ignore

    # Adjust some columns of the dataframe we just constructed
    obs_log = obs_log.rename(columns={"DATE-END": "DATE_END", "DATE-OBS": "DATE_OBS"})

    # convert the date columns from string to Time type so we can easily compute mid time
    obs_log["DATE_OBS"] = obs_log["DATE_OBS"].apply(lambda t: Time(t))
    obs_log["DATE_END"] = obs_log["DATE_END"].apply(lambda t: Time(t))

    # add middle of observation time
    dts = (obs_log["DATE_END"] - obs_log["DATE_OBS"]) / 2
    obs_log["MID_TIME"] = obs_log["DATE_OBS"] + dts

    # track which extension in the fits files the images are
    obs_log["EXTENSION"] = extension_list

    # the filename the extension was pulled from
    obs_log["FITS_FILENAME"] = processed_filname_list

    # version of UVOT2FITS
    obs_log["CREATOR"] = creator_list

    # obs_log["IMAGE_SHAPE_ROWS"] = image_shape_row_list
    # obs_log["IMAGE_SHAPE_COLS"] = image_shape_col_list

    # translates horizons results (left) to observation log column names (right)
    # documentation of values returned by Horizons available at
    # https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides
    ephemeris_info = {
        # Target heliocentric distance, float, in AU
        "r": "HELIO",
        # Target heliocentric distance change rate, float, in km/s
        "r_rate": "HELIO_V",
        # Target distance from observation point (@swift in our case), float, in AU
        "delta": "OBS_DIS",
        # Target solar phase angle, float, degrees (Sun-Target-Object angle)
        "alpha": "PHASE",
        # Target right ascension, float, degrees
        "RA": "RA",
        # Target declination, float, degrees
        "DEC": "DEC",
    }
    # make dataframe with columns of the ephemeris_info values
    horizon_dataframe = pd.DataFrame(columns=list(ephemeris_info.values()))  # type: ignore

    horizons_progress_bar = tqdm(obs_log["MID_TIME"], unit="observations")

    # for each row, query Horizons for our object at 'mid_time' and fill the dataframe with response info
    for k, mid_time in enumerate(horizons_progress_bar):
        horizons_progress_bar.set_description(
            f"Horizons querying {obs_log['OBS_ID'][k]} extension {obs_log['EXTENSION'][k]} ..."
        )

        horizons_response = Horizons(
            id=horizons_id, location="@swift", epochs=mid_time.jd, id_type="designation"
        )
        eph = horizons_response.ephemerides(closest_apparition=True)  # type: ignore
        # append this row of information to our horizons dataframe
        horizon_dataframe.loc[len(horizon_dataframe.index)] = [
            eph[x][0] for x in ephemeris_info.keys()
        ]
        horizons_response._session.close()

    obs_log = pd.concat([obs_log, horizon_dataframe], axis=1)

    x_list = []
    y_list = []
    # use the positions found from Horizons to find the pixel center of the comet based on its image WCS
    for i, (ra, dec) in enumerate(zip(obs_log["RA"], obs_log["DEC"])):
        x, y = wcs_list[i].wcs_world2pix(ra, dec, 1)
        x_list.append(float(x))
        y_list.append(float(y))

    obs_log["PX"] = x_list
    obs_log["PY"] = y_list

    # convert columns to their respective types
    obs_log["FILTER"] = obs_log["FILTER"].astype(str).map(obs_string_to_filter)

    obs_log["OBS_ID"] = obs_log["OBS_ID"].apply(swift_observation_id_from_int)
    obs_log["ORBIT_ID"] = obs_log["OBS_ID"].apply(swift_orbit_id_from_obsid)

    obs_log["DATAMODE"] = obs_log.DATAMODE.apply(datamode_from_fits_keyword_string)
    obs_log["ARCSECS_PER_PIXEL"] = obs_log.DATAMODE.apply(datamode_to_pixel_resolution)

    # Conversion rate of 1 pixel to km: DATAMODE now holds image resolution in arcseconds/pixel
    obs_log["KM_PER_PIX"] = obs_log.apply(
        lambda row: (
            (
                ((2 * np.pi) / (3600.0 * 360.0))
                * row.ARCSECS_PER_PIXEL
                * row.OBS_DIS
                * u.AU  # type: ignore
            ).to_value(
                u.km  # type: ignore
            )
        ),
        axis=1,
    )

    obs_log["manual_veto"] = False * len(obs_log.index)

    # initialize user-specified comet centers as invalid
    obs_log["USER_CENTER_X"] = [invalid_user_center_value()] * len(obs_log.index)
    obs_log["USER_CENTER_Y"] = [invalid_user_center_value()] * len(obs_log.index)

    return obs_log
