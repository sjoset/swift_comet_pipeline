#!/usr/bin/env python3

import os
import glob
import pathlib
import sys
import logging as log

from argparse import ArgumentParser

from configs import read_swift_project_config
from swift_data import SwiftData

from epochs import read_epoch, write_epoch
from manual_veto import manual_veto
from user_input import get_yes_no, get_selection


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


def select_epoch(epoch_dir: pathlib.Path) -> pathlib.Path:
    glob_pattern = str(epoch_dir / pathlib.Path("*.parquet"))

    epoch_filename_list = sorted(glob.glob(glob_pattern))

    epoch_path = pathlib.Path(epoch_filename_list[get_selection(epoch_filename_list)])

    return epoch_path


def main():
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None or swift_project_config.observation_log is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    epoch_dir_path = swift_project_config.epoch_dir_path
    if epoch_dir_path is None:
        print(f"Could not find epoch_path in {swift_project_config_path}, exiting.")
        return

    epoch_path = select_epoch(epoch_dir_path)
    epoch_pre_veto = read_epoch(epoch_path)

    epoch_post_veto = manual_veto(
        swift_data=swift_data, epoch=epoch_pre_veto, epoch_title=epoch_path.stem
    )

    print("Save epoch?")
    save_epoch = get_yes_no()
    if not save_epoch:
        return

    write_epoch(epoch=epoch_post_veto, epoch_path=epoch_path)


if __name__ == "__main__":
    sys.exit(main())
