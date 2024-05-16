import numpy as np
from icecream import ic
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import astropy.units as u

from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig


def download_orbital_data(
    swift_project_config: SwiftProjectConfig,
    time_before_first_observation: u.Quantity = 1 * u.year,
    time_after_last_observation: u.Quantity = 1 * u.year,
) -> None:
    # TODO: document this function

    # TODO: this should just return dataframes for earth and comet, or product objects
    pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    data_ingestion_files.observation_log.read()
    obs_log = data_ingestion_files.observation_log.data
    if obs_log is None:
        ic("Could not read observation log!")
        return

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log["MID_TIME"])) - time_before_first_observation
    time_stop = Time(np.max(obs_log["MID_TIME"])) + time_after_last_observation

    # make our dictionary for the horizons query
    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    if data_ingestion_files.comet_orbital_data.exists():
        print("Comet data already found!")
        # TODO: ask to delete and re-do
        # TODO: this shouldn't return here: we might need the earth orbital data
        return

    # location=None defaults to solar system barycenter
    comet_horizons_response = Horizons(
        id=swift_project_config.jpl_horizons_id,
        location=None,
        id_type="designation",
        epochs=epochs,
    )

    # get comet orbital data in a horizons response and put it in a pandas dataframe
    comet_vectors = comet_horizons_response.vectors(closest_apparition=True)  # type: ignore
    comet_df = comet_vectors.to_pandas()
    data_ingestion_files.comet_orbital_data.data = comet_df
    data_ingestion_files.comet_orbital_data.write()

    # Same process for earth over the time frame of our comet data
    earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
    earth_vectors = earth_horizons_response.vectors()  # type: ignore
    earth_df = earth_vectors.to_pandas()

    # TODO: this should check for earth before it writes
    data_ingestion_files.earth_orbital_data.data = earth_df
    data_ingestion_files.earth_orbital_data.write()
