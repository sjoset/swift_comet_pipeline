from dataclasses import dataclass

from swift_comet_pipeline.scp_types.compound.vectorial_model_fit import (
    VectorialModelFit,
)
from swift_comet_pipeline.scp_types.primitive.column_density import ColumnDensity
from swift_comet_pipeline.scp_types.serialization.structure_unstructure import (
    from_unstructured,
    to_unstructured,
)


@dataclass
class RadialProfileWaterProductionAnalysis:
    near_fit: VectorialModelFit
    far_fit: VectorialModelFit
    full_fit: VectorialModelFit

    oh_column_density: ColumnDensity


def radial_profile_water_production_analysis_from_json(
    json_dict: dict,
) -> RadialProfileWaterProductionAnalysis:
    return from_unstructured(obj=json_dict, c=RadialProfileWaterProductionAnalysis)


def json_from_radial_profile_water_production_analysis(
    rpwpa: RadialProfileWaterProductionAnalysis,
) -> dict:
    return to_unstructured(obj=rpwpa)
