#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log

from argparse import ArgumentParser

from configs import read_swift_project_config
from pipeline_files import PipelineFiles
from swift_data import SwiftData

from manual_veto import manual_veto
from tui import get_yes_no, epoch_menu


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

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    if pipeline_files.epoch_products is None:
        print("No epoch files found! Exiting.")
        return 0
    epoch_product = epoch_menu(pipeline_files)
    if epoch_product is None:
        print("Could not select epoch, exiting.")
        return 1
    epoch_product.load_product()
    epoch_pre_veto = epoch_product.data_product

    epoch_post_veto = manual_veto(
        swift_data=swift_data,
        epoch=epoch_pre_veto,
        epoch_title=epoch_product.product_path.stem,
    )

    print("Save epoch?")
    save_epoch = get_yes_no()
    if not save_epoch:
        return 0

    epoch_product.data_product = epoch_post_veto
    epoch_product.save_product()


if __name__ == "__main__":
    sys.exit(main())
