#!/usr/bin/env python3

import sys
import os
import pathlib
import numpy as np
import logging as log
import astropy.units as u

from argparse import ArgumentParser
from astroquery.jplhorizons import Horizons
from astropy.time import Time

from configs import read_swift_project_config
from pipeline_files import PipelineFiles


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config",
        nargs="?",
        help="Filename of project config",
        default="config.yaml",
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
    """
    Takes an observation log and outputs the orbital data vectors for the object by querying jplhorizons, covering
    a year before the first observation and a year after the last observation
    """
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(
        base_product_save_path=swift_project_config.product_save_path
    )
    if pipeline_files.observation_log is None:
        print("Observation log not found! Exiting.")
        return 1

    pipeline_files.observation_log.load_product()
    obs_log = pipeline_files.observation_log.data_product

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log["MID_TIME"])) - 1 * u.year  # type: ignore
    time_stop = Time(np.max(obs_log["MID_TIME"])) + 1 * u.year  # type: ignore

    # make our dictionary for the horizons query
    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    if not pipeline_files.comet_orbital_data.product_path.exists():
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

        pipeline_files.comet_orbital_data.data_product = comet_df
        pipeline_files.comet_orbital_data.save_product()
        print(
            f"Comet orbital data saved to {pipeline_files.comet_orbital_data.product_path}"
        )
    else:
        print(
            f"Skipped comet orbital data because {pipeline_files.comet_orbital_data.product_path} exists."
        )

    if not pipeline_files.earth_orbital_data.product_path.exists():
        # Same process for earth over the time frame of our comet data
        earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
        earth_vectors = earth_horizons_response.vectors()  # type: ignore
        earth_df = earth_vectors.to_pandas()

        pipeline_files.earth_orbital_data.data_product = earth_df
        pipeline_files.earth_orbital_data.save_product()
        print(
            f"Earth orbital data saved to {pipeline_files.earth_orbital_data.product_path}"
        )
    else:
        print(
            f"Skipped earth orbital data because {pipeline_files.earth_orbital_data.product_path} exists."
        )


if __name__ == "__main__":
    sys.exit(main())
