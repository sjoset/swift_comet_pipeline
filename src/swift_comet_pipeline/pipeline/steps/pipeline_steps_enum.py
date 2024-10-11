from enum import StrEnum, auto


class SwiftCometPipelineStepEnum(StrEnum):

    # data ingestion
    observation_log = auto()
    download_orbital_data = auto()
    identify_epochs = auto()
    veto_images = auto()

    # epoch subpipeline
    epoch_stack = auto()
    determine_background = auto()
    background_subtract = auto()
    aperture_analysis = auto()
    vectorial_analysis = auto()

    # results
    build_lightcurves = auto()
