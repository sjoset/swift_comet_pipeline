import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import astropy.units as u
from rich import print as rprint

from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.tui import wait_for_key

from swift_comet_pipeline.pipeline_files import PipelineFiles, PipelineProductType


def pipeline_extra_orbital_data(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(
        base_product_save_path=swift_project_config.product_save_path
    )

    obs_log = pipeline_files.read_pipeline_product(
        p=PipelineProductType.observation_log
    )

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log["MID_TIME"])) - 1 * u.year  # type: ignore
    time_stop = Time(np.max(obs_log["MID_TIME"])) + 1 * u.year  # type: ignore
    rprint(
        f"Requesting orbit data from [blue]{time_start}[/blue] to [yellow]{time_stop}[/yellow] ..."
    )

    # make our dictionary for the horizons query
    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    if not pipeline_files.exists(p=PipelineProductType.comet_orbital_data):
        # location=None defaults to solar system barycenter
        comet_horizons_response = Horizons(
            id=swift_project_config.jpl_horizons_id,
            location=None,
            id_type="designation",
            epochs=epochs,
        )

        rprint("[blue]Querying Horizons for comet orbit data ...[/blue]")
        # get comet orbital data in a horizons response and put it in a pandas dataframe
        comet_vectors = comet_horizons_response.vectors(closest_apparition=True)  # type: ignore
        comet_df = comet_vectors.to_pandas()
        pipeline_files.write_pipeline_product(
            p=PipelineProductType.comet_orbital_data, data=comet_df
        )
        rprint(
            f"[green]Comet orbital data saved to {pipeline_files.comet_orbital_data_path}[/green]"
        )
    else:
        rprint("[red]Comet orbital data exists, skipping ...")

    if not pipeline_files.exists(PipelineProductType.earth_orbital_data):
        rprint("[blue]Querying Horizons for earth orbit data ...[/blue]")
        # Same process for earth over the time frame of our comet data
        earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
        earth_vectors = earth_horizons_response.vectors()  # type: ignore
        earth_df = earth_vectors.to_pandas()

        pipeline_files.write_pipeline_product(
            PipelineProductType.earth_orbital_data, data=earth_df
        )
        rprint(
            f"[green]Earth orbital data saved to {pipeline_files.earth_orbital_data_path}[/green]"
        )
    else:
        rprint("[red]Earth orbital data exists, skipping ...")

    wait_for_key()
