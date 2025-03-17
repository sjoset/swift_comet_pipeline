import pathlib
from dataclasses import dataclass

from swift_comet_pipeline.modeling.vectorial_model_backend import VectorialModelBackend
from swift_comet_pipeline.modeling.vectorial_model_grid import VectorialModelGridQuality


# TODO: document
@dataclass
class SwiftProjectConfig:
    """
    Holds configuration data for the current comet analysis project
    """

    swift_data_path: pathlib.Path
    jpl_horizons_id: str
    project_path: pathlib.Path
    vectorial_model_quality: VectorialModelGridQuality
    vectorial_model_backend: VectorialModelBackend
    vectorial_fitting_requires_km: float
    near_far_split_radius_km: float
