from enum import StrEnum, auto


class PipelineSteps(StrEnum):
    observation_log = auto()
    identify_epochs = auto()
    veto_epoch = auto()
    epoch_stacking = auto()
    background_analysis = auto()
    qH2O_vs_aperture_radius = auto()
    qH2O_from_profile = auto()

    exit_pipeline = auto()

    @classmethod
    def all_pipeline_steps(cls):
        return [x for x in cls]
