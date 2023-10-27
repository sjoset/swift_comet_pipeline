#!/usr/bin/env python3

import os
import pathlib
import sys
import logging as log
import numpy as np
from argparse import ArgumentParser


from swift_comet_pipeline.configs import read_swift_project_config
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.pipeline_steps import (
    menu_stack_all_or_selection,
    print_stacked_images_summary,
)
from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.stacking import uw1_and_uvv_stacks_from_epoch
from swift_comet_pipeline.tui import epoch_menu


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
    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config)
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print("Error reading config file {swift_project_config_path}, exiting.")
        return 1
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    is_epoch_stacked = print_stacked_images_summary(pipeline_files=pipeline_files)
    if all(is_epoch_stacked.values()):
        print("Everything stacked! Nothing to do.")
        return 0

    menu_selection = menu_stack_all_or_selection()

    epochs_to_stack = []
    ask_to_save_stack = True
    show_stacked_images = True
    if menu_selection == "a":
        epochs_to_stack = pipeline_files.epoch_products
        ask_to_save_stack = False
        show_stacked_images = False
    elif menu_selection == "s":
        epochs_to_stack = [epoch_menu(pipeline_files)]

    if epochs_to_stack is None:
        print("Pipeline error! This is a bug with pipeline_files.epoch_products!")
        return 1

    for epoch_product in epochs_to_stack:
        if epoch_product is None:
            print("Error selecting epoch! Exiting.")
            return 1
        epoch_product.load_product()
        epoch = epoch_product.data_product

        non_vetoed_epoch = epoch[epoch.manual_veto == np.False_]

        uw1_and_uvv_stacks_from_epoch(
            pipeline_files=pipeline_files,
            swift_data=swift_data,
            epoch=non_vetoed_epoch,
            epoch_path=epoch_product.product_path,
            do_coincidence_correction=True,
            ask_to_save_stack=ask_to_save_stack,
            show_stacked_images=show_stacked_images,
        )


if __name__ == "__main__":
    sys.exit(main())
