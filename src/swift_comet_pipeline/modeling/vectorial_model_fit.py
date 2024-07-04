from dataclasses import dataclass

import numpy as np
import astropy.units as u
from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult
from scipy.optimize import curve_fit

from swift_comet_pipeline.comet.column_density import ColumnDensity


@dataclass
class VectorialModelFit:
    # water production that best matches comet column density
    best_fit_Q: u.Quantity
    # err of the associated fit
    best_fit_Q_err: u.Quantity
    # column density resulting from this fit
    vectorial_column_density: ColumnDensity


def fit_vectorial_model_to_comet_column_density(
    comet_column_density: ColumnDensity,
    vmr: VectorialModelResult,
    model_Q: u.Quantity,
    r_fit_min: u.Quantity,  # type: ignore
    r_fit_max: u.Quantity,  # type: ignore
) -> tuple[u.Quantity, float]:
    # TODO: documentation

    # vectorial model column density interpolation is in 1/m^2, with radii in meters, so convert here
    ccd_fit = (comet_column_density.cd_cm2 / u.cm**2).to(1 / u.m**2).value  # type: ignore
    rs_fit = (comet_column_density.rs_km * u.km).to(u.m).value  # type: ignore

    fit_mask_min = rs_fit > r_fit_min.to_value(u.m)  # type: ignore
    fit_mask_max = rs_fit < r_fit_max.to_value(u.m)  # type: ignore
    fit_mask = np.logical_and(fit_mask_min, fit_mask_max)
    ccd_fit = ccd_fit[fit_mask]
    rs_fit = rs_fit[fit_mask]

    def vcd_func(r: float, q_ratio: float) -> float:
        return q_ratio * vmr.column_density_interpolation(r)  # type: ignore

    popt, pcov = curve_fit(vcd_func, rs_fit, ccd_fit)

    best_fit_production = model_Q * popt[0]
    fit_err = np.sqrt(pcov[0][0])

    return (best_fit_production, fit_err)


def vectorial_fit(
    comet_column_density: ColumnDensity,
    vmr: VectorialModelResult,
    model_Q: u.Quantity,
    r_fit_min: u.Quantity,
    r_fit_max: u.Quantity,
) -> VectorialModelFit:
    # TODO: documentation

    fit_Q, fit_err = fit_vectorial_model_to_comet_column_density(
        comet_column_density=comet_column_density,
        vmr=vmr,
        model_Q=model_Q,
        r_fit_min=r_fit_min,
        r_fit_max=r_fit_max,
    )

    q_ratio = (fit_Q / model_Q).decompose()

    vec_col_dens_m2 = vmr.column_density_interpolation(
        (comet_column_density.rs_km * u.km).to_value(u.m)  # type: ignore
    )
    vec_col_dens = ColumnDensity(
        rs_km=comet_column_density.rs_km,
        cd_cm2=q_ratio
        * (vec_col_dens_m2 / (u.m**2)).to_value(1 / u.cm**2),  # type: ignore
    )

    return VectorialModelFit(
        best_fit_Q=fit_Q,
        best_fit_Q_err=fit_err * fit_Q,
        vectorial_column_density=vec_col_dens,
    )
