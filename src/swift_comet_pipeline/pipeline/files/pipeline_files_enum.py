from enum import StrEnum, auto


class PipelineFilesEnum(StrEnum):

    # data ingestion
    observation_log = auto()
    comet_orbital_data = auto()
    earth_orbital_data = auto()
    epoch_pre_stack = auto()

    # subpipeline: stacking step
    epoch_post_stack = auto()
    stacked_image = auto()
    exposure_map = auto()

    # subpipeline: background determination
    background_determination = auto()

    # subpipeline: background subtracted images
    background_subtracted_image = auto()

    # subpipeline: aperture analysis
    aperture_analysis = auto()

    # subpipeline: vectorial analysis
    extracted_profile = auto()
    extracted_profile_image = auto()
    median_subtracted_image = auto()
    median_divided_image = auto()

    # lightcurve results
    aperture_lightcurve = auto()
    bayesian_aperture_lightcurve = auto()
    complete_vectorial_lightcurve = auto()
    # bayesian_vectorial_lightcurve = auto()
    best_near_fit_vectorial_lightcurve = auto()
    best_far_fit_vectorial_lightcurve = auto()
    best_full_fit_vectorial_lightcurve = auto()


def is_data_ingestion_file(pfe: PipelineFilesEnum) -> bool:
    if pfe in [
        PipelineFilesEnum.observation_log,
        PipelineFilesEnum.comet_orbital_data,
        PipelineFilesEnum.earth_orbital_data,
        # PipelineFilesEnum.epoch_pre_stack,
    ]:
        return True
    else:
        return False


def is_analysis_result_file(pfe: PipelineFilesEnum) -> bool:
    if pfe in [
        PipelineFilesEnum.aperture_lightcurve,
        PipelineFilesEnum.bayesian_aperture_lightcurve,
        PipelineFilesEnum.complete_vectorial_lightcurve,
        # PipelineFilesEnum.bayesian_vectorial_lightcurve,
        PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
        PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
        PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
    ]:
        return True
    else:
        return False


def is_epoch_subpipeline_file(pfe: PipelineFilesEnum) -> bool:
    if not is_data_ingestion_file(pfe) and not is_analysis_result_file(pfe):
        return True
    else:
        return False
