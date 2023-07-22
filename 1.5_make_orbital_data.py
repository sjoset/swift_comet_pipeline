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

from configs import read_swift_project_config, write_swift_project_config
from observation_log import read_observation_log


__version__ = "0.0.1"


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config", nargs=1, help="Filename of project config"
    )
    parser.add_argument(
        "--comet",
        "-c",
        default="horizons_comet_orbital_data.csv",
        help="Filename to store comet orbital data (csv format)",
    )
    parser.add_argument(
        "--earth",
        "-e",
        default="horizons_earth_orbital_data.csv",
        help="Filename to store earth orbital data (csv format)",
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

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None or swift_project_config.observation_log is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    horizon_id = swift_project_config.jpl_horizons_id

    obs_log = read_observation_log(swift_project_config.observation_log)

    # take a time range of a year before the first observation to a year after the last
    time_start = Time(np.min(obs_log["MID_TIME"])) - 1 * u.year  # type: ignore
    time_stop = Time(np.max(obs_log["MID_TIME"])) + 1 * u.year  # type: ignore

    epochs = {"start": time_start.iso, "stop": time_stop.iso, "step": "1d"}

    # location=None defaults to solar system barycenter
    comet_horizons_response = Horizons(
        id=horizon_id, location=None, id_type="smallbody", epochs=epochs
    )

    # get comet orbital data in a horizons response and put it in a pandas dataframe
    comet_vectors = comet_horizons_response.vectors()  # type: ignore
    comet_df = comet_vectors.to_pandas()

    comet_vectors_output_path = (
        swift_project_config.product_save_path.expanduser().resolve()
        / pathlib.Path(args.comet)
    )

    comet_df.to_csv(comet_vectors_output_path)
    print(f"Output successfully written to {comet_vectors_output_path}")

    # Same process for earth over the time frame of our comet data
    earth_horizons_response = Horizons(id=399, location=None, epochs=epochs)
    earth_vectors = earth_horizons_response.vectors()  # type: ignore
    earth_df = earth_vectors.to_pandas()
    earth_vectors_output_path = (
        swift_project_config.product_save_path.expanduser().resolve()
        / pathlib.Path(args.earth)
    )
    earth_df.to_csv(earth_vectors_output_path)
    print(f"Output successfully written to {comet_vectors_output_path}")

    # update the project config
    swift_project_config.comet_orbital_data_path = comet_vectors_output_path
    swift_project_config.earth_orbital_data_path = earth_vectors_output_path
    write_swift_project_config(
        config_path=pathlib.Path(swift_project_config_path),
        swift_project_config=swift_project_config,
    )


if __name__ == "__main__":
    sys.exit(main())
