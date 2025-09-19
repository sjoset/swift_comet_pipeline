import itertools

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from tqdm import tqdm


from swift_comet_pipeline.scp_types.primitive import *

from swift_comet_pipeline.data_ingestion.observation_log.fits_header_extraction import (
    level_2_observation_to_series,
)
from swift_comet_pipeline.data_ingestion.observation_log.comet_center_tracking import (
    invalid_user_center_value,
)
from swift_comet_pipeline.scp_types.compound.swift_dataset import SwiftDataset
from swift_comet_pipeline.swift.filters.uvot_filter_to_string import (
    obs_string_to_filter,
)
from swift_comet_pipeline.swift.swift_observation_id import (
    swift_observation_id_from_int,
)
from swift_comet_pipeline.swift.swift_orbit_id import swift_orbit_id_from_obsid
from swift_comet_pipeline.swift.uvot_datamodes import (
    datamode_from_fits_keyword_string,
    datamode_to_pixel_resolution,
)


def build_observation_log(
    swift_data: SwiftDataset,
    horizons_id: str,
) -> SwiftUvotObservationLogDataframe | None:
    """
    include observation ids that have images that:
        - are from uvot
            - are data mode in sky_units (sk), any filter
            OR
            - are event mode, any filter
    and returns an observation log in the form of a pandas dataframe
    """

    observation_entries_list = []

    observation_progress_bar = tqdm(swift_data.observation_ids, unit="observations")
    for obsid in observation_progress_bar:

        # For this observation ID, get images in every filter - we could limit it, but
        # we can include all of the observations in the log and filter later if we want
        # just certain filters
        all_observations_for_this_obsid = [
            swift_data.observations[obsid, ft] for ft in UvotFilter.all_filters()
        ]
        # filter out the ones that were not found
        all_observations_for_this_obsid = list(
            filter(lambda x: x is not None, all_observations_for_this_obsid)
        )
        # flatten this list into 1d
        all_observations_for_this_obsid = list(
            itertools.chain.from_iterable(all_observations_for_this_obsid)  # type: ignore
        )

        if len(all_observations_for_this_obsid) == 0:
            print(
                f"No valid UVOT observations found for observation ID {obsid}, skipping..."
            )

            continue

        observations_this_obsid = [
            level_2_observation_to_series(obs=x)
            for x in all_observations_for_this_obsid
            if x is not None
        ]

        valid_observations_this_obsid = list(
            itertools.chain.from_iterable(observations_this_obsid)  # type: ignore
        )
        observation_entries_list.append(valid_observations_this_obsid)

        observation_progress_bar.set_description(f"Observation ID: {obsid}")

    # This list is built by looping observation ids, which may contain sub-lists of multiple observations
    flattened_observation_series_list = list(
        itertools.chain.from_iterable(observation_entries_list)
    )
    obs_log = pd.DataFrame(flattened_observation_series_list)

    # Adjust some columns of the dataframe we just constructed
    obs_log = obs_log.rename(columns={"DATE-END": "DATE_END", "DATE-OBS": "DATE_OBS"})

    # convert the date columns from string to Time type so we can easily compute mid time
    obs_log["DATE_OBS"] = obs_log["DATE_OBS"].apply(lambda t: Time(t))
    obs_log["DATE_END"] = obs_log["DATE_END"].apply(lambda t: Time(t))

    # add middle of observation time
    dts = (obs_log["DATE_END"] - obs_log["DATE_OBS"]) / 2
    obs_log["MID_TIME"] = obs_log["DATE_OBS"] + dts

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
        # Rate of change of RA in arcseconds per hour
        "RA_rate": "RA_RATE",
        # Rate of change of Dec in arcseconds per hour
        "DEC_rate": "DEC_RATE",
        # direction of sky motion, position angle, degrees
        "velocityPA": "SKY_MOTION_PA",
        # ion tail position angle, degrees
        "sunTargetPA": "ION_TAIL_PA",
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

    # convert arcseconds per hour to arcseconds per minute
    sky_motion_conversion_factor = 1.0 / 60.0
    horizon_dataframe.RA_RATE *= sky_motion_conversion_factor
    horizon_dataframe.DEC_RATE *= sky_motion_conversion_factor

    horizon_dataframe["SKY_MOTION"] = np.hypot(
        horizon_dataframe.RA_RATE, horizon_dataframe.DEC_RATE
    )

    obs_log = pd.concat([obs_log, horizon_dataframe], axis=1)

    # # TODO: remove x_list, y_list
    # x_list = []
    # y_list = []
    # # use the positions found from Horizons to find the pixel center of the comet based on its image WCS
    # for ra, dec, wcs_cur in zip(obs_log["RA"], obs_log["DEC"], obs_log["WCS"]):
    #     x, y = wcs_cur.wcs_world2pix(ra, dec, 1)
    #     x_list.append(float(x))
    #     y_list.append(float(y))
    # print(f"Old comet x list: {x_list}")

    comet_centers = [
        img_wcs.wcs_world2pix(r, d, 0)
        for img_wcs, r, d in zip(obs_log.WCS, obs_log.RA, obs_log.DEC)
    ]
    comet_center_xs = [float(x[0]) for x in comet_centers]
    comet_center_ys = [float(x[1]) for x in comet_centers]

    obs_log["PX"] = comet_center_xs
    obs_log["PY"] = comet_center_ys

    # convert columns to their respective types
    obs_log["FILTER"] = obs_log["FILTER"].astype(str).map(obs_string_to_filter)

    obs_log["OBS_ID"] = obs_log["OBS_ID"].apply(swift_observation_id_from_int)
    obs_log["ORBIT_ID"] = obs_log["OBS_ID"].apply(swift_orbit_id_from_obsid)

    # obs_log["DATAMODE"] = obs_log.DATAMODE.apply(datamode_from_fits_keyword_string)
    # print(f"Full paths:")
    # for i, row in obs_log.iterrows():
    #     print(f"{i}: {row.FULL_FITS_PATH}, {row.OBS_ID}, {row.FITS_FILENAME}")

    obs_log["DATAMODE"] = obs_log.apply(
        lambda row: datamode_from_fits_keyword_string(
            datamode=row.DATAMODE, fits_file_path=row.FULL_FITS_PATH
        ),
        axis=1,
    )
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

    obs_log["epoch_id"] = "" * len(obs_log.index)

    # drop this column now that we are done with it
    obs_log = obs_log.drop("WCS", axis=1)

    return obs_log
