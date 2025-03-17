from functools import cache
import pathlib
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.types.fluorescence_g_factor import FluorescenceGFactor1AU


@cache
def read_hydroxyl_gfactor_1au_data(
    fluorescence_file: pathlib.Path,
) -> FluorescenceGFactor1AU:
    df = pd.read_csv(fluorescence_file)
    helio_vs = np.array(df["heliocentric_v_kms"].values)

    # add up channels and attach units as described by file
    gfactors = ((df["0-0"] + df["1-0"] + df["1-1"]).values) * 1e-16

    return FluorescenceGFactor1AU(helio_vs, gfactors)


# TODO: change helio_v_kms to be an astropy quantity, add decorator to enforce proper input
# TODO: make rh a parameter and rename function to gfactor: scale the 1AU gfactor data by 1/(rh_in_au)**2
@cache
def hydroxyl_gfactor_1au(
    helio_v_kms: float, fluorescence_data: FluorescenceGFactor1AU | None = None
) -> float:
    if fluorescence_data is None:
        spc = read_swift_pipeline_config()
        if spc is None:
            print("Could not read swift pipeline config!")
            exit(1)
        fluorescence_data = read_hydroxyl_gfactor_1au_data(
            fluorescence_file=spc.oh_fluorescence_path
        )

    # TODO: cite the source of the g factor data
    g1au_interpolation = interp1d(
        fluorescence_data.helio_vs, fluorescence_data.gfactors, kind="cubic"
    )

    return g1au_interpolation(helio_v_kms)
