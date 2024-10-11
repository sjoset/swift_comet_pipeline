from dataclasses import dataclass
from enum import StrEnum, auto

from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)


class SwiftCometPipelineStepStatus(StrEnum):
    complete = auto()
    partial = auto()
    not_complete = auto()
    invalid = auto()


@dataclass(frozen=True)
class SwiftCometPipelineStep:
    step: SwiftCometPipelineStepEnum
    require: SwiftCometPipelineStepEnum | None


@dataclass(frozen=True)
class ObservationLogStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.observation_log
    require: SwiftCometPipelineStepEnum | None = None


@dataclass(frozen=True)
class DownloadOrbitalDataStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.download_orbital_data
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.observation_log


@dataclass(frozen=True)
class IdentifyEpochsStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.identify_epochs
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.observation_log


@dataclass(frozen=True)
class VetoImagesStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.veto_images
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.identify_epochs


@dataclass(frozen=True)
class EpochStackStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.epoch_stack
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.identify_epochs


@dataclass(frozen=True)
class DetermineBackgroundStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.determine_background
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.epoch_stack


@dataclass(frozen=True)
class BackgroundSubtractStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.background_subtract
    require: SwiftCometPipelineStepEnum = (
        SwiftCometPipelineStepEnum.determine_background
    )


@dataclass(frozen=True)
class ApertureAnalysisStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.aperture_analysis
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.background_subtract


@dataclass(frozen=True)
class VectorialAnalysisStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.vectorial_analysis
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.background_subtract


# TODO: this should depend on both vectorial and aperture analysis
@dataclass(frozen=True)
class BuildLightcurvesStep(SwiftCometPipelineStep):
    step: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.build_lightcurves
    require: SwiftCometPipelineStepEnum = SwiftCometPipelineStepEnum.vectorial_analysis
