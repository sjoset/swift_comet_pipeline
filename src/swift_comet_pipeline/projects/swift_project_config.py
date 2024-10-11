import pathlib
from dataclasses import dataclass

from swift_comet_pipeline.modeling.vectorial_model_backend import VectorialModelBackend
from swift_comet_pipeline.modeling.vectorial_model_grid import VectorialModelGridQuality


@dataclass
class SwiftProjectConfig:
    swift_data_path: pathlib.Path
    jpl_horizons_id: str
    project_path: pathlib.Path
    vectorial_model_quality: VectorialModelGridQuality
    vectorial_model_backend: VectorialModelBackend
