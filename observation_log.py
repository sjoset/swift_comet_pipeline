import itertools
import pathlib
import logging as log
import pandas as pd
import pyarrow as pa
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons

from typing import List, Optional


from tqdm import tqdm

from swift_types import (
    SwiftData,
    SwiftObservationID,
    SwiftFilter,
    SwiftObservationLog,
    filter_to_obs_string,
    swift_observation_id_from_int,
    obs_string_to_filter,
    swift_orbit_id_from_obsid,
)


__all__ = [
    "observation_log_schema",
    "build_observation_log",
    "read_observation_log",
    "write_observation_log",
    "includes_uvv_and_uw1_filters",
]


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
            pa.field("RA", pa.float64()),
            pa.field("DEC", pa.float64()),
            pa.field("PX", pa.float64()),
            pa.field("PY", pa.float64()),
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
    # list of extensions that describe which sub-image inside a FITS file we are describing
    extension_list = []
    # list of world coordinate systems of each image
    wcs_list = []
    # keep a list of the filenames that the extensions come from
    processed_filname_list = []

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

                    # add a row to the dataframe from the header info
                    obs_log.loc[len(obs_log.index)] = [
                        header[i] for i in fits_header_entries_to_read
                    ]

                    wcs_list.append(WCS(header))
                    extension_list.append(ext_id)
                    processed_filname_list.append(image_path.name)  # type: ignore

                    image_progress_bar.set_description(f"Image: {image_path.name}")  # type: ignore

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
    horizon_dataframe = pd.DataFrame(columns=list(ephemeris_info.values()))

    horizons_progress_bar = tqdm(obs_log["MID_TIME"], unit="observations")

    # for each row, query Horizons for our object at 'mid_time' and fill the dataframe with response info
    for k, mid_time in enumerate(horizons_progress_bar):
        horizons_response = Horizons(
            id=horizons_id, location="@swift", epochs=mid_time.jd, id_type="designation"
        )
        eph = horizons_response.ephemerides(closest_apparition=True)  # type: ignore
        # append this row of information to our horizon dataframe
        horizon_dataframe.loc[len(horizon_dataframe.index)] = [
            eph[x][0] for x in ephemeris_info.keys()
        ]
        horizons_response._session.close()

        horizons_progress_bar.set_description(
            f"Horizons querying {obs_log['OBS_ID'][k]} extension {obs_log['EXTENSION'][k]}"
        )

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

    obs_log[["DATE_OBS", "DATE_END", "MID_TIME"]] = obs_log[
        ["DATE_OBS", "DATE_END", "MID_TIME"]
    ].apply(pd.to_datetime)

    obs_log["FILTER"] = obs_log["FILTER"].astype(str).map(obs_string_to_filter)

    obs_log["OBS_ID"] = obs_log["OBS_ID"].apply(swift_observation_id_from_int)
    obs_log["ORBIT_ID"] = obs_log["OBS_ID"].apply(swift_orbit_id_from_obsid)

    return obs_log


def write_observation_log(
    obs_log: SwiftObservationLog,
    obs_log_path: pathlib.Path,
    additional_schema: pa.lib.Schema = None,
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

    schema = observation_log_schema()
    if additional_schema is not None:
        schema = pa.unify_schemas([schema, additional_schema])

    oc.to_parquet(obs_log_path, schema=schema)


def includes_uvv_and_uw1_filters(
    obs_log: SwiftObservationLog,
) -> bool:
    """
    To find OH and perform dust subtraction we need data from the UV and UW1 filter from somewhere across the given data set in orbit_ids.
    Returns a list of orbits that have UV or UW1 images, after removing orbits that have no data in the UV or UW1 filters
    """

    has_uvv_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uvv]
    has_uvv_set = set(has_uvv_filter["ORBIT_ID"])

    has_uw1_filter = obs_log[obs_log["FILTER"] == SwiftFilter.uw1]
    has_uw1_set = set(has_uw1_filter["ORBIT_ID"])

    has_both = len(has_uvv_set) > 0 and len(has_uw1_set) > 0

    # print(
    #     f"Found {len(has_uw1_filter)} uw1 observations and {len(has_uvv_filter)} uvv observations"
    # )
    # contributing_orbits = has_uvv_set
    # contributing_orbits.update(has_uw1_set)

    # return (has_both, list(contributing_orbits))
    return has_both


# def print_parquet_metadata():
#
#     m = pa.parquet.read_metadata(parquet_path)
#     print(m.metadata[b'metadata_key'])


# def get_obsids_in_orbits(
#     obs_log: SwiftObservationLog, orbit_ids: List[SwiftOrbitID]
# ) -> List[SwiftObservationID]:
#     """Returns a list of all the observation ids contained in the given orbit_ids"""
#     ml = match_by_orbit_ids_and_filters(
#         obs_log=obs_log, orbit_ids=orbit_ids, filter_types=SwiftFilter.all_filters()
#     )
#
#     return sorted(np.unique(ml["OBS_ID"].values))
#
#
# def match_by_obsids_and_filters(
#     obs_log: SwiftObservationLog,
#     obsids: List[SwiftObservationID],
#     filter_types: List[SwiftFilter],
# ) -> SwiftObservationLog:
#     """Returns the matching rows of the observation log that match any combination of the given obsids & filters"""
#     masks = []
#     for obsid, filter_type in itertools.product(obsids, filter_types):
#         masks.append((obs_log["FILTER"] == filter_type) & (obs_log["OBS_ID"] == obsid))
#
#     mask = np.logical_or.reduce(masks)
#
#     return obs_log[mask]
#
#
# def match_by_orbit_ids_and_filters(
#     obs_log: SwiftObservationLog,
#     orbit_ids: List[SwiftOrbitID],
#     filter_types: List[SwiftFilter],
# ) -> SwiftObservationLog:
#     """Returns the matching rows of the observation log that match any combination of the given orbit_ids & filters"""
#     masks = []
#     for orbit_id, filter_type in itertools.product(orbit_ids, filter_types):
#         masks.append(
#             (obs_log["ORBIT_ID"] == orbit_id) & (obs_log["FILTER"] == filter_type)
#         )
#
#     mask = np.logical_or.reduce(masks)
#     return obs_log[mask]
