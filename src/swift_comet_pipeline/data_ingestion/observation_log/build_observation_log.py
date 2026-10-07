import itertools
import logging as log

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


def query_horizons_comet_ephemerides(
    horizons_id: str, epoch_mid_times: list[Time], horizons_batch_size: int = 20
) -> pd.DataFrame:

    ephemeris_info = {
        "r": "HELIO",
        "r_rate": "HELIO_V",
        "delta": "OBS_DIS",
        "alpha": "PHASE",
        "RA": "RA",
        "DEC": "DEC",
        "RA_rate": "RA_RATE",
        "DEC_rate": "DEC_RATE",
        "velocityPA": "SKY_MOTION_PA",
        "sunTargetPA": "ION_TAIL_PA",
    }

    if horizons_batch_size < 1:
        raise ValueError("horizons_batch_size must be at least 1.")

    epochs_jd = np.asarray([epoch.jd for epoch in epoch_mid_times])

    if not np.all(np.diff(epochs_jd) > 0):
        raise ValueError("Horizons epochs must be unique and chronologically ordered.")

    batches = np.array_split(
        epochs_jd,
        np.arange(horizons_batch_size, len(epochs_jd), horizons_batch_size),
    )

    frames: list[pd.DataFrame] = []

    for requested_jd in tqdm(batches, unit="Horizons queries"):
        eph = Horizons(
            id=horizons_id,
            location="@swift",
            epochs=requested_jd.tolist(),
        ).ephemerides(
            quantities="1,3,19,20,24,27",
        )

        returned_jd = np.asarray(eph["datetime_jd"], dtype=float)

        if len(returned_jd) != len(requested_jd):
            raise RuntimeError(
                f"Horizons returned {len(returned_jd)} rows "
                f"for {len(requested_jd)} requested epochs."
            )

        if not np.allclose(
            returned_jd,
            requested_jd,
            rtol=0,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Horizons returned epochs that do not match " "the requested epochs."
            )

        frames.append(
            eph[list(ephemeris_info)].to_pandas().rename(columns=ephemeris_info)
        )

    return pd.concat(frames, ignore_index=True)


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
            log.info(
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
    obs_log: pd.DataFrame = pd.DataFrame(flattened_observation_series_list)

    # Adjust some columns of the dataframe we just constructed
    obs_log = obs_log.rename(columns={"DATE-END": "DATE_END", "DATE-OBS": "DATE_OBS"})

    # convert the date columns from string to Time type so we can easily compute mid time
    obs_log["DATE_OBS"] = obs_log["DATE_OBS"].apply(lambda t: Time(t))
    obs_log["DATE_END"] = obs_log["DATE_END"].apply(lambda t: Time(t))

    # add middle of observation time
    dts = (obs_log["DATE_END"] - obs_log["DATE_OBS"]) / 2
    obs_log["MID_TIME"] = obs_log["DATE_OBS"] + dts

    # chronologically sort the observation log: Horizons batch lookups will sort by time, so we want our data in chronological order to match
    # for when we concat the horizons dataframe and the obs_log
    obs_log = obs_log.sort_values("MID_TIME").reset_index()

    horizon_dataframe: pd.DataFrame = query_horizons_comet_ephemerides(
        horizons_id=horizons_id, epoch_mid_times=obs_log.MID_TIME.to_list()
    )

    # convert arcseconds per hour to arcseconds per minute
    sky_motion_conversion_factor = 1.0 / 60.0
    horizon_dataframe.RA_RATE *= sky_motion_conversion_factor
    horizon_dataframe.DEC_RATE *= sky_motion_conversion_factor

    horizon_dataframe["SKY_MOTION"] = np.hypot(
        horizon_dataframe.RA_RATE, horizon_dataframe.DEC_RATE
    )

    obs_log = pd.concat([obs_log, horizon_dataframe], axis=1)

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
