#!/usr/bin/env python3

import os
import pathlib
import sys
import warnings
import logging as log
from enum import StrEnum

from astropy.wcs.wcs import FITSFixedWarning
from argparse import ArgumentParser

from swift_comet_pipeline.modeling.vectorial_model import vectorial_model_settings_init
from swift_comet_pipeline.projects.configs import read_or_create_project_config
from swift_comet_pipeline.tui.pipeline_extras import pipeline_extras_menu
from swift_comet_pipeline.tui.pipeline_steps_background_analysis_step import (
    background_analysis_step,
)
from swift_comet_pipeline.tui.pipeline_steps_generate_lightcurve import (
    generate_lightcurve_step,
)
from swift_comet_pipeline.tui.pipeline_steps_observation_log import (
    observation_log_step,
)
from swift_comet_pipeline.tui.pipeline_steps_identify_epochs import identify_epochs_step
from swift_comet_pipeline.tui.pipeline_steps_qH2O_from_profile import (
    qH2O_from_profile_step,
)
from swift_comet_pipeline.tui.pipeline_steps_qH2O_vs_aperture_radius import (
    qH2O_vs_aperture_radius_step,
)
from swift_comet_pipeline.tui.pipeline_steps_vectorial_profile_fitting import (
    vectorial_fitting_step,
)
from swift_comet_pipeline.tui.pipeline_steps_veto_epoch import veto_epoch_step
from swift_comet_pipeline.tui.pipeline_steps_epoch_stacking import epoch_stacking_step
from swift_comet_pipeline.tui.tui_common import (
    clear_screen,
    get_selection,
)


class PipelineStepsMenuEntry(StrEnum):
    observation_log = "generate observation log"
    identify_epochs = "slice observation log into epochs"
    veto_epoch = "view and veto images in epoch"
    epoch_stacking = "stack images in an epoch"
    background_analysis = "background analysis and subtraction"
    qH2O_vs_aperture_radius = "water production as a function of aperture radius"
    qH2O_from_profile = "water production from a comet profile/slice"
    vectorial_fitting = (
        "derive water production from comet profile via fitting to model"
    )
    generate_lightcurve = "generate lightcurve"

    extra_functions = "extra functions"

    @classmethod
    def all_pipeline_steps(cls):
        return [x for x in cls]


def process_args():
    # Parse command-line arguments
    parser = ArgumentParser(
        usage="%(prog)s [options]",
        description=__doc__,
        prog=os.path.basename(sys.argv[0]),
    )
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


# TODO: menu with icons that indicate whether step has been done completely or partially
def pipeline_step_menu() -> PipelineStepsMenuEntry:
    return PipelineStepsMenuEntry.observation_log


def main():
    # we don't care about these particular warnings
    warnings.resetwarnings()
    warnings.filterwarnings("ignore", category=FITSFixedWarning, append=True)

    args = process_args()
    swift_project_config_path = pathlib.Path(args.swift_project_config)

    swift_project_config = read_or_create_project_config(
        swift_project_config_path=swift_project_config_path
    )

    if swift_project_config is None:
        print("Could not load a valid configuration! Exiting.")
        return

    # set up the cache db and other stuff
    vectorial_model_settings_init(swift_project_config=swift_project_config)

    exit_program = False
    pipeline_steps = PipelineStepsMenuEntry.all_pipeline_steps()
    while not exit_program:
        clear_screen()
        step_selection = get_selection(pipeline_steps)
        if step_selection is None:
            exit_program = True
            continue
        step = pipeline_steps[step_selection]
        # TODO: could this be a Protocol as a PipelineStep(swift_project_config) -> ret val
        # along with a next_step(), prev_step(), and whether this one has been completed
        #  maybe similar thing for epoch subpipelines
        if step == PipelineStepsMenuEntry.observation_log:
            observation_log_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.identify_epochs:
            identify_epochs_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.veto_epoch:
            veto_epoch_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.epoch_stacking:
            epoch_stacking_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.background_analysis:
            background_analysis_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.qH2O_vs_aperture_radius:
            qH2O_vs_aperture_radius_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.qH2O_from_profile:
            qH2O_from_profile_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.vectorial_fitting:
            vectorial_fitting_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.generate_lightcurve:
            generate_lightcurve_step(swift_project_config=swift_project_config)
        elif step == PipelineStepsMenuEntry.extra_functions:
            pipeline_extras_menu(swift_project_config=swift_project_config)
        else:
            exit_program = True


if __name__ == "__main__":
    sys.exit(main())
