import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import astropy.units as u

from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.projects.configs import SwiftProjectConfig


# TODO: add dataclass for these orbital vector dataframes


def download_orbital_data(
    swift_project_config: SwiftProjectConfig,
    time_before_first_observation: u.Quantity = 1 * u.year,  # type: ignore
    time_after_last_observation: u.Quantity = 1 * u.year,  # type: ignore
) -> None:
    # TODO: document this function

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # # TODO: should this just return dataframes for earth and comet, or product objects?
    # pipeline_files = PipelineFiles(project_path=swift_project_config.project_path)
    # data_ingestion_files = pipeline_files.data_ingestion_files

    obs_log = scp.get_product_data(pf=PipelineFilesEnum.observation_log)
    assert obs_log is not None

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log["MID_TIME"])) - time_before_first_observation
    time_stop = Time(np.max(obs_log["MID_TIME"])) + time_after_last_observation

    # print(f"Downloading orbital data from {time_start.ymdhms} to {time_stop.ymdhms}")

    # make our dictionary for the horizons query
    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    if scp.exists(pf=PipelineFilesEnum.comet_orbital_data):
        print("Comet orbital data already found!")
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

    comet_product = scp.get_product(pf=PipelineFilesEnum.comet_orbital_data)
    assert comet_product is not None
    comet_product.data = comet_df
    comet_product.write()

    if scp.exists(pf=PipelineFilesEnum.earth_orbital_data):
        print("Earth orbital data already found!")
        return

    # Same process for earth over the time frame of our comet data
    earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
    earth_vectors = earth_horizons_response.vectors()  # type: ignore
    earth_df = earth_vectors.to_pandas()

    earth_product = scp.get_product(pf=PipelineFilesEnum.earth_orbital_data)
    assert earth_product is not None
    earth_product.data = earth_df
    earth_product.write()
