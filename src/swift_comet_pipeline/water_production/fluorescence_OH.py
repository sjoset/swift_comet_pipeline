from functools import cache
import pathlib
import numpy as np
import pandas as pd
import astropy.units as u

from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import TypeAlias

from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.water_production.flux_OH import OHFlux

NumOH: TypeAlias = ValueAndStandardDev

# TODO:
# https://asteroid.lowell.edu/comet/gfactor


# TODO: add entry to identify the molecule this gfactor is describing or rename this for hydroxyl
@dataclass(frozen=True)
class FluorescenceGFactor1AU:
    helio_vs: np.ndarray
    gfactors: np.ndarray


@cache
def read_gfactor_1au_data(fluorescence_file: pathlib.Path) -> FluorescenceGFactor1AU:
    # TODO: rename this function to reflect that our file reflects values computed for OH
    df = pd.read_csv(fluorescence_file)
    helio_vs = np.array(df["heliocentric_v_kms"].values)

    # add up channels and attach units as described by file
    gfactors = ((df["0-0"] + df["1-0"] + df["1-1"]).values) * 1e-16

    return FluorescenceGFactor1AU(helio_vs, gfactors)


# TODO: change helio_v_kms to be an astropy quantity, add decorator to enforce proper input
# TODO: make rh a parameter and rename function to gfactor: scale the 1AU gfactor data by 1/(rh_in_au)**2
@cache
def gfactor_1au(
    helio_v_kms: float, fluorescence_data: FluorescenceGFactor1AU | None = None
) -> float:
    if fluorescence_data is None:
        spc = read_swift_pipeline_config()
        if spc is None:
            print("Could not read swift pipeline config!")
            exit(1)
        fluorescence_data = read_gfactor_1au_data(
            fluorescence_file=spc.oh_fluorescence_path
        )

    # TODO: cite the source of the g factor data
    g1au_interpolation = interp1d(
        fluorescence_data.helio_vs, fluorescence_data.gfactors, kind="cubic"
    )

    return g1au_interpolation(helio_v_kms)


# TODO: change helio_v_kms, helio_r_au, delta_au to be an astropy quantity, add decorator to enforce proper input
def flux_OH_to_num_OH(
    flux_OH: OHFlux,
    helio_r_au: float,
    helio_v_kms: float,
    delta_au: float,
    fluorescence_data: FluorescenceGFactor1AU | None = None,
) -> NumOH:
    # g factors given in terms of ergs, so we need to use cm while calculating luminescence
    delta = (delta_au * u.AU).to_value(u.cm)  # type: ignore
    luminescence = 4 * np.pi * flux_OH.value * delta**2
    luminescence_err = flux_OH.sigma * 4 * np.pi * delta**2

    # g_factor = g1au_interpolation(helio_v_kms) / (helio_r_au**2)
    g_factor = gfactor_1au(
        helio_v_kms=helio_v_kms, fluorescence_data=fluorescence_data
    ) / (helio_r_au**2)

    num_OH = luminescence / g_factor
    num_err = luminescence_err / g_factor

    return NumOH(value=num_OH, sigma=num_err)
