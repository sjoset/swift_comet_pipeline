from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive.column_density import ColumnDensity


@dataclass
class VectorialModelFit:
    # water production that best matches comet column density
    best_fit_q_per_s: float
    # err of the associated fit
    best_fit_q_per_s_err: float
    # column density resulting from this fit
    vectorial_column_density: ColumnDensity
