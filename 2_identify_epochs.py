#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log

from argparse import ArgumentParser

from configs import read_swift_project_config, write_swift_project_config
from swift_filter import SwiftFilter

from observation_log import read_observation_log, includes_uvv_and_uw1_filters
from epochs import write_epoch
from pipeline_files import epoch_filenames_from_epoch_list, get_default_epoch_dir
from user_input import get_yes_no
from epoch_time_window import (
    epochs_from_time_delta,
    select_epoch_time_window,
)


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options] [inputfile] [outputfile]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="increase verbosity level"
    )
    parser.add_argument(
        "swift_project_config", nargs=1, help="Filename of project config"
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
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None or swift_project_config.observation_log is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    obs_log = read_observation_log(swift_project_config.observation_log)

    if not includes_uvv_and_uw1_filters(obs_log=obs_log):
        print("The selection does not have data in both uw1 and uvv filters!")

    # only uw1 and uvv filters
    filter_mask = (obs_log["FILTER"] == SwiftFilter.uw1) | (
        obs_log["FILTER"] == SwiftFilter.uvv
    )
    obs_log = obs_log[filter_mask]

    dt = select_epoch_time_window(obs_log=obs_log)
    epoch_list = epochs_from_time_delta(obs_log=obs_log, max_time_between_obs=dt)

    print("Save epochs?")
    save_epochs = get_yes_no()
    if not save_epochs:
        return

    epoch_dir = get_default_epoch_dir(
        product_save_path=swift_project_config.product_save_path
    )
    epoch_dir.mkdir(parents=True, exist_ok=True)
    # construct full paths for each epoch and write them out
    epoch_names = epoch_filenames_from_epoch_list(epoch_list)
    for filename, epoch in zip(epoch_names, epoch_list):
        full_path = epoch_dir / filename
        print(f"Writing file: {full_path}")
        write_epoch(epoch=epoch, epoch_path=full_path)

    # update project config with the epoch directory, and save it back to the file
    swift_project_config.epoch_dir_path = epoch_dir
    write_swift_project_config(
        config_path=pathlib.Path(swift_project_config_path),
        swift_project_config=swift_project_config,
    )


if __name__ == "__main__":
    sys.exit(main())
