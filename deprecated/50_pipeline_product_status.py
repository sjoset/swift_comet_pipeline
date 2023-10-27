#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log

# import pyarrow as pa
# import numpy as np
# import astropy.units as u

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking import StackingMethod

from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.configs import read_swift_project_config
from swift_comet_pipeline.observation_log import (
    build_observation_log,
    # observation_log_schema,
    write_observation_log,
)
from swift_comet_pipeline.swift_filter import SwiftFilter


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options]",
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
    parser.add_argument(
        "--output",
        "-o",
        default="observation_log.parquet",
        help="Filename of observation log output",
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


# def bool_to_x_or_check(x: bool):
#     if x:
#         return "✔"
#     else:
#         return "✗"


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()

    swift_project_config_path = pathlib.Path(args.swift_project_config[0])
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return 1

    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    print(
        pipeline_files.observation_log.product_path,
        bool_to_x_or_check(pipeline_files.observation_log.product_path.exists()),
    )
    print(
        pipeline_files.comet_orbital_data.product_path,
        bool_to_x_or_check(pipeline_files.comet_orbital_data.product_path.exists()),
    )
    print(
        pipeline_files.earth_orbital_data.product_path,
        bool_to_x_or_check(pipeline_files.earth_orbital_data.product_path.exists()),
    )

    if pipeline_files.epoch_products is None:
        print("No epochs defined yet!")
        exit(0)

    # print(pipeline_files.epoch_file_paths)
    print("Epochs:")
    for x in pipeline_files.epoch_products:
        print(x.product_path.stem)

    print("")
    print("Stacked epochs and images:")
    for x in pipeline_files.epoch_file_paths:  # type: ignore
        ep_prod = pipeline_files.stacked_epoch_products[x]  # type: ignore
        print(ep_prod.product_path, bool_to_x_or_check(x.exists()))
        uw1_sum = pipeline_files.stacked_image_products[  # type: ignore
            x, SwiftFilter.uw1, StackingMethod.summation
        ]
        print(
            f"uw1 sum: {uw1_sum.product_path.stem}, {bool_to_x_or_check(uw1_sum.exists())}"
        )
        if uw1_sum.exists():
            uw1_sum.load_product()
            print("Dimensions:", uw1_sum.data_product.data.shape)

        bg_prod = pipeline_files.analysis_background_products[x]  # type: ignore
        print(bg_prod.product_path, bool_to_x_or_check(bg_prod.exists()))
        if bg_prod.exists():
            bg_prod.load_product()
            print(bg_prod.data_product["method"])

        uw1_bg_sub = pipeline_files.analysis_bg_subtracted_images[x, SwiftFilter.uw1, StackingMethod.summation]  # type: ignore
        print(uw1_bg_sub.product_path, bool_to_x_or_check(uw1_bg_sub.exists()))
        if uw1_bg_sub.exists():
            uw1_bg_sub.load_product()
            print("Dimensions:", uw1_bg_sub.data_product.data.shape)
        uvv_bg_sub = pipeline_files.analysis_bg_subtracted_images[x, SwiftFilter.uvv, StackingMethod.summation]  # type: ignore
        print(uvv_bg_sub.product_path, bool_to_x_or_check(uvv_bg_sub.exists()))
        if uvv_bg_sub.exists():
            uvv_bg_sub.load_product()
            print("Dimensions:", uvv_bg_sub.data_product.data.shape)

        q = pipeline_files.analysis_qh2o_products[x]  # type: ignore
        print(q.product_path, bool_to_x_or_check(q.exists()))
        if q.exists():
            q.load_product()
            print(q.data_product.Q_H2O[0])

        print("")

    # print(pipeline_files.stacked_image_product_dict)

    # horizons_id = swift_project_config.jpl_horizons_id
    # sdd = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    # df = build_observation_log(
    #     swift_data=sdd,
    #     obsids=sdd.get_all_observation_ids(),
    #     horizons_id=horizons_id,
    # )

    # if df is None:
    #     print(
    #         "Could not construct the observation log in memory, exiting without writing output."
    #     )
    #     return 1
    #
    # write_observation_log(df, pipeline_files.get_observation_log_path())


if __name__ == "__main__":
    sys.exit(main())
