#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log

from argparse import ArgumentParser

from configs import read_swift_project_config

from swift_filter import SwiftFilter

from observation_log import includes_uvv_and_uw1_filters

from pipeline_files import EpochProduct, PipelineFiles
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
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    # obs_log = read_observation_log(pipeline_files.get_observation_log_path())
    pipeline_files.observation_log.load_product()
    obs_log = pipeline_files.observation_log.data_product

    if not includes_uvv_and_uw1_filters(obs_log=obs_log):
        print("The selection does not have data in both uw1 and uvv filters!")

    # only show uw1 and uvv filters on timeline
    filter_mask = (obs_log["FILTER"] == SwiftFilter.uw1) | (
        obs_log["FILTER"] == SwiftFilter.uvv
    )
    filtered_obs_log = obs_log[filter_mask]

    # filtered_obs_log = obs_log

    dt = select_epoch_time_window(obs_log=filtered_obs_log)
    epoch_list = epochs_from_time_delta(
        obs_log=filtered_obs_log, max_time_between_obs=dt
    )

    print("Save epochs?")
    save_epochs = get_yes_no()
    if not save_epochs:
        return

    epoch_path_list = pipeline_files.determine_epoch_file_paths(epoch_list=epoch_list)
    for epoch, epoch_path in zip(epoch_list, epoch_path_list):
        epoch_product = EpochProduct(product_path=epoch_path)
        epoch_product.data_product = epoch
        print(f"Writing {epoch_product.product_path} ...")
        epoch_product.save_product()


if __name__ == "__main__":
    sys.exit(main())
