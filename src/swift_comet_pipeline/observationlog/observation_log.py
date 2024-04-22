import itertools
import pathlib
from typing import List, Optional, TypeAlias

import logging as log
import numpy as np
import pandas as pd
import pyarrow as pa
import astropy.units as u

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from tqdm import tqdm

from swift_comet_pipeline.swift.swift_data import (
    SwiftData,
    SwiftObservationID,
    swift_observation_id_from_int,
    swift_orbit_id_from_obsid,
)
from swift_comet_pipeline.swift.swift_filter import (
    SwiftFilter,
    obs_string_to_filter,
    filter_to_obs_string,
)
from swift_comet_pipeline.swift.uvot_image import (
    PixelCoord,
    datamode_from_fits_keyword_string,
    datamode_to_pixel_resolution,
    float_to_pixel_resolution,
)


SwiftObservationLog: TypeAlias = pd.DataFrame


def invalid_user_center_value() -> float:
    return -1


def has_valid_user_center_value(row: pd.Series) -> bool:

    if (
        row.USER_CENTER_X != invalid_user_center_value()
        and row.USER_CENTER_Y != invalid_user_center_value()
    ):
        return True
    else:
        return False


def get_horizons_comet_center(row: pd.Series) -> PixelCoord:
    return PixelCoord(x=row.PX, y=row.PY)


def get_user_specified_comet_center(row: pd.Series) -> Optional[PixelCoord]:
    if has_valid_user_center_value(row=row):
        return PixelCoord(x=row.USER_CENTER_X, y=row.USER_CENTER_Y)
    else:
        return None


# TODO: add documentation for each of these entries and what they hold
def observation_log_schema() -> pa.lib.Schema:
    schema = pa.schema(
        [
            pa.field("OBS_ID", pa.int64()),
            pa.field("DATE_OBS", pa.string()),
            pa.field("DATE_END", pa.string()),
            pa.field("MID_TIME", pa.string()),
            pa.field("FILTER", pa.string()),
            pa.field("PA_PNT", pa.float64()),
            pa.field("RA_OBJ", pa.float64()),
            pa.field("DEC_OBJ", pa.float64()),
            pa.field("EXPOSURE", pa.float64()),
            pa.field("EXTENSION", pa.int16()),
            pa.field("FITS_FILENAME", pa.string()),
            pa.field("HELIO", pa.float64()),
            pa.field("HELIO_V", pa.float64()),
            pa.field("OBS_DIS", pa.float64()),
            pa.field("PHASE", pa.float64()),
            # coordinates given by Horizons for the center of the comet
            pa.field("RA", pa.float64()),
            pa.field("DEC", pa.float64()),
            # PX, PY: pixel x and y coordinates of comet center after converting RA, DEC using WCS
            pa.field("PX", pa.float64()),
            pa.field("PY", pa.float64()),
            # x and y coordinates of comet center as selected by user during veto step
            pa.field("USER_CENTER_X", pa.float64()),
            pa.field("USER_CENTER_Y", pa.float64()),
            # string as taken from the FITS header keyword DATAMODE
            pa.field("DATAMODE", pa.string()),
            # how many arcseconds per pixel in this data mode
            pa.field("ARCSECS_PER_PIXEL", pa.float64()),
            # conversion from pixels to kilometers, based on the distance of the comet from the observation point
            pa.field("KM_PER_PIX", pa.float64()),
            # TODO: figure out if we really need to keep these
            # # number of rows in the image (height)
            # pa.field("IMAGE_SHAPE_ROWS", pa.int16()),
            # # number of columns in the image (width)
            # pa.field("IMAGE_SHAPE_COLS", pa.int16()),
            pa.field("CREATOR", pa.string()),
            pa.field("manual_veto", pa.bool_()),
        ]
    )

    return schema


def build_observation_log(
    swift_data: SwiftData,
    obsids: List[SwiftObservationID],
    horizons_id: str,
) -> Optional[SwiftObservationLog]:
    """
    Takes a list of observation ids and looks through all images that:
        - are from uvot
        - are sky_units (sk)
        - are in any filter
    and returns an observation log in the form of a pandas dataframe
    """

    all_filters = SwiftFilter.all_filters()

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
    ephemeris_info = {
        "r": "HELIO",
        "r_rate": "HELIO_V",
        "delta": "OBS_DIS",
        "alpha": "PHASE",
        "RA": "RA",
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
                * u.AU
            ).to_value(u.km)
        ),
        axis=1,
    )

    obs_log["manual_veto"] = False * len(obs_log.index)

    # initialize these as invalid
    # obs_log["USER_CENTER_X"] = [-1] * len(obs_log.index)
    # obs_log["USER_CENTER_Y"] = [-1] * len(obs_log.index)
    obs_log["USER_CENTER_X"] = [invalid_user_center_value()] * len(obs_log.index)
    obs_log["USER_CENTER_Y"] = [invalid_user_center_value()] * len(obs_log.index)

    return obs_log


def read_observation_log(
    obs_log_path: pathlib.Path, additional_schema: pa.lib.Schema = None
) -> SwiftObservationLog:
    """
    Reads an observation log generated by build_observation_log and converts column data types where needed
    """
    schema = observation_log_schema()
    if additional_schema is not None:
        schema = pa.unify_schemas([schema, additional_schema])

    obs_log = pd.read_parquet(obs_log_path, schema=schema)

    # post-processing on the columns to get them into a datatype we want: we have to store these as strings
    # but datetime objects are much nicer to work with
    obs_log[["DATE_OBS", "DATE_END", "MID_TIME"]] = obs_log[
        ["DATE_OBS", "DATE_END", "MID_TIME"]
    ].apply(pd.to_datetime)

    # filters are stored as strings as well, convert them to SwiftFilter objects
    obs_log.FILTER = obs_log["FILTER"].astype(str).map(obs_string_to_filter)

    # observation ID is stored as an int: convert to SwiftObservationID and SwiftOrbitID types
    obs_log.OBS_ID = obs_log["OBS_ID"].apply(swift_observation_id_from_int)
    obs_log["ORBIT_ID"] = obs_log["OBS_ID"].apply(swift_orbit_id_from_obsid)

    # DATAMODE is stored as a string: convert to valid SwiftImageDataMode
    obs_log.DATAMODE = obs_log.DATAMODE.apply(datamode_from_fits_keyword_string)

    # ARCSECS_PER_PIXEL is stored as a float: convert to valid SwiftPixelResolution
    obs_log.ARCSECS_PER_PIXEL = obs_log.ARCSECS_PER_PIXEL.apply(
        float_to_pixel_resolution
    )

    return obs_log


def write_observation_log(
    obs_log: SwiftObservationLog,
    obs_log_path: pathlib.Path,
    # additional_schema: pa.lib.Schema = None,
) -> None:
    """
    Copy obs_log and process the columns to data types that fit our schema, then save
    """
    oc = obs_log.copy()

    oc[["DATE_OBS", "DATE_END", "MID_TIME"]] = oc[
        ["DATE_OBS", "DATE_END", "MID_TIME"]
    ].astype(str)

    oc["OBS_ID"] = oc["OBS_ID"].astype(int)
    oc["FILTER"] = oc["FILTER"].map(filter_to_obs_string)

    oc.DATAMODE = oc.DATAMODE.astype(str)
    oc.ARCSECS_PER_PIXEL = oc.ARCSECS_PER_PIXEL.astype(float)

    # the rest of the column conversions should be taken care of automatically through type hints from the schema

    schema = observation_log_schema()
    # if additional_schema is not None:
    #     schema = pa.unify_schemas([schema, additional_schema])

    oc.to_parquet(obs_log_path, schema=schema)


def includes_uvv_and_uw1_filters(
    obs_log: SwiftObservationLog,
) -> bool:
    has_uvv_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uvv]
    has_uvv_set = set(has_uvv_filter["ORBIT_ID"])

    has_uw1_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uw1]
    has_uw1_set = set(has_uw1_filter["ORBIT_ID"])

    has_both = len(has_uvv_set) > 0 and len(has_uw1_set) > 0

    return has_both


def get_image_path_from_obs_log_row(swift_data: SwiftData, obs_log_row) -> pathlib.Path:
    image_path = (
        swift_data.get_uvot_image_directory(obsid=obs_log_row.OBS_ID)
        / obs_log_row.FITS_FILENAME
    )

    return image_path


def get_image_from_obs_log_row(swift_data: SwiftData, obs_log_row):
    image_path = get_image_path_from_obs_log_row(
        swift_data=swift_data, obs_log_row=obs_log_row
    )
    image_data = fits.getdata(image_path, ext=obs_log_row.EXTENSION)

    return image_data


def get_header_from_obs_log_row(swift_data: SwiftData, obs_log_row):
    image_path = (
        swift_data.get_uvot_image_directory(obsid=obs_log_row.OBS_ID)
        / obs_log_row.FITS_FILENAME
    )
    header = fits.getheader(image_path)

    return header


# def print_parquet_metadata():
#
#     m = pa.parquet.read_metadata(parquet_path)
#     print(m.metadata[b'metadata_key'])
