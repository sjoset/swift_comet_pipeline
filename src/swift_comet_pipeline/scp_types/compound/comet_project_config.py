import pathlib
from dataclasses import dataclass, field

from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter
from swift_comet_pipeline.scp_types.primitive.vectorial_model_backend import (
    VectorialModelBackend,
)
from swift_comet_pipeline.scp_types.primitive.vectorial_model_grid_quality import (
    VectorialModelGridQuality,
)


@dataclass
class CometProjectConfig:
    """
    Holds configuration data for the current comet analysis project
    """

    # path to downloaded swift data - see SwiftData class for folder structure expected
    swift_data_path: pathlib.Path
    # how to query horizons for the comet ephemera
    jpl_horizons_id: str
    # where to store the analysis
    project_path: pathlib.Path = pathlib.Path.cwd()

    # vectorial model settings
    vectorial_model_quality: VectorialModelGridQuality = VectorialModelGridQuality.high
    vectorial_model_backend: VectorialModelBackend = VectorialModelBackend.sbpy
    vectorial_fitting_requires_km: float = 100000
    near_far_split_radius_km: float = 50000

    stacking_methods: list[StackingMethod] = field(
        default_factory=lambda: [StackingMethod.summation]
    )

    oh_filters: list[UvotFilter] = field(
        default_factory=lambda: [UvotFilter.uw1, UvotFilter.uw2, UvotFilter.uuu]
    )
    dust_filters: list[UvotFilter] = field(
        default_factory=lambda: [
            UvotFilter.uvv,
            UvotFilter.uuu,
            UvotFilter.ubb,
            UvotFilter.um2,
        ]
    )

    # which dust rednesses do we use for computation?
    dust_redness_min: DustReddeningPercent = 0.0
    dust_redness_max: DustReddeningPercent = 40.0
    dust_redness_step: float = 1.0

    # TODO: add entries to specify dust redness mean and sigma for q expectation values?

    # midpoint of the uw1 and uvv filters as the default for redness context
    redness_mid_wavelength_nm: float = 438.181
