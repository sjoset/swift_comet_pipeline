from enum import StrEnum, auto


class ProductKind(StrEnum):

    # data ingestion
    observation_log = auto()
    sliced_observation_log = auto()
    comet_orbital_data = auto()
    earth_orbital_data = auto()

    # subpipeline: stacking step

    # images produced by per-filter stacking
    stacked_image = auto()
    # exposure_map = auto()

    # # subpipeline: background determination
    # background_determination = auto()
    #
    # # subpipeline: background subtracted images
    # background_subtracted_stacked_image = auto()
    #
    # # subpipeline: aperture analysis
    # aperture_analysis = auto()
    #
    # # subpipeline: vectorial analysis
    # extracted_profile = auto()
    # extracted_profile_image = auto()
    # median_subtracted_image = auto()
    # median_divided_image = auto()
    #
    # # lightcurve results
    # aperture_lightcurve = auto()
    # bayesian_aperture_lightcurve = auto()
    # complete_vectorial_lightcurve = auto()
    # # bayesian_vectorial_lightcurve = auto()
    # best_near_fit_vectorial_lightcurve = auto()
    # best_far_fit_vectorial_lightcurve = auto()
    # best_full_fit_vectorial_lightcurve = auto()
    # unified_lightcurve = auto()
