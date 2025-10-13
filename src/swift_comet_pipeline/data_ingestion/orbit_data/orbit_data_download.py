from astroquery.jplhorizons import Horizons
import numpy as np
import pandas as pd
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.pipeline.product_system.registry_and_store import Products


def orbit_data_download(
    scp: Products,
    horizons_id: str | None = None,
    time_before_first_observation: u.Quantity = 1 * u.year,  # type: ignore
    time_after_last_observation: u.Quantity = 1 * u.year,  # type: ignore
) -> pd.DataFrame | None:
    """
    Return Horizons orbit information for the comet with the given horizons id, or earth orbit if id is None
    """
    # TODO: decouple this from scp and instead take a time window as arguments

    obs_log_df = scp.load_obs_log()
    if obs_log_df is None:
        return None

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log_df["MID_TIME"])) - time_before_first_observation
    time_stop = Time(np.max(obs_log_df["MID_TIME"])) + time_after_last_observation

    # print(f"Downloading orbital data from {time_start.ymdhms} to {time_stop.ymdhms}")

    if horizons_id is None:
        return earth_orbit_data_download(time_start=time_start, time_stop=time_stop)

    return comet_orbit_data_download(
        time_start=time_start, time_stop=time_stop, horizons_id=horizons_id
    )


def comet_orbit_data_download(
    time_start: Time, time_stop: Time, horizons_id: str
) -> pd.DataFrame:

    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    # location=None defaults to solar system barycenter
    comet_horizons_response = Horizons(
        id=horizons_id,
        location=None,
        id_type="designation",
        epochs=epochs,
    )

    # get comet orbital data in a horizons response and put it in a pandas dataframe
    comet_vectors = comet_horizons_response.vectors(closest_apparition=True)  # type: ignore
    comet_df = comet_vectors.to_pandas()

    return comet_df


def earth_orbit_data_download(time_start: Time, time_stop: Time) -> pd.DataFrame:

    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    # Same process for earth over the time frame of our comet data
    earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
    earth_vectors = earth_horizons_response.vectors()  # type: ignore
    earth_df = earth_vectors.to_pandas()

    return earth_df
