import pathlib
from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive.vectorial_model_backend import (
    VectorialModelBackend,
)
from swift_comet_pipeline.scp_types.primitive.vectorial_model_grid_quality import (
    VectorialModelGridQuality,
)


# TODO: document
@dataclass
class CometProjectConfig:
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
