import pathlib
import numpy as np
import pandas as pd
import astropy.units as u

from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional

from configs import read_swift_pipeline_config


__all__ = ["FluorescenceGFactor1AU", "read_gfactor_1au", "flux_OH_to_num_OH"]


@dataclass
class FluorescenceGFactor1AU:
    helio_vs: np.ndarray
    gfactors: np.ndarray


def read_gfactor_1au(fluorescence_file: pathlib.Path) -> FluorescenceGFactor1AU:
    df = pd.read_csv(fluorescence_file)
    helio_vs = df["heliocentric_v_kms"].values

    # add up channels and attach units as described by file
    gfactors = ((df["0-0"] + df["1-0"] + df["1-1"]).values) * 1e-16

    return FluorescenceGFactor1AU(helio_vs, gfactors)


def flux_OH_to_num_OH(
    flux_OH: float,
    helio_r_au: float,
    helio_v_kms: float,
    delta_au: float,
    fluorescence_data: Optional[FluorescenceGFactor1AU] = None,
) -> float:
    if fluorescence_data is None:
        spc = read_swift_pipeline_config()
        if spc is None:
            return 0
        fluorescence_data = read_gfactor_1au(fluorescence_file=spc.oh_fluorescence_path)

    g1au_interpolation = interp1d(
        fluorescence_data.helio_vs, fluorescence_data.gfactors, kind="cubic"
    )

    # g factors given in terms of ergs, so we need to use cm while calculating luminescence
    delta = (delta_au * u.AU).to_value(u.cm)  # type: ignore
    luminescence = 4 * np.pi * flux_OH * delta**2

    g_factor = g1au_interpolation(helio_v_kms) / (helio_r_au**2)
    # print(
    #     f"g factor at {helio_r_au} AU, heliocentric velocity {helio_v_kms}: {g_factor}"
    # )

    num_OH = luminescence / g_factor

    return num_OH
