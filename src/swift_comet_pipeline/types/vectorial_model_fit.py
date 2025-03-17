from dataclasses import dataclass

import astropy.units as u

from swift_comet_pipeline.types.column_density import ColumnDensity


@dataclass
class VectorialModelFit:
    # water production that best matches comet column density
    best_fit_Q: u.Quantity
    # err of the associated fit
    best_fit_Q_err: u.Quantity
    # column density resulting from this fit
    vectorial_column_density: ColumnDensity
