from functools import cache
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.types.oh_fluorescence_g_factor import OHFluorescenceGFactor1AU


@cache
def read_hydroxyl_gfactor_1au_data() -> OHFluorescenceGFactor1AU:
    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read swift pipeline config!")
        exit(1)

    fluorescence_file = spc.oh_fluorescence_path

    df = pd.read_csv(fluorescence_file)

    helio_vs = np.array(df["v_kms"].values)

    # add up channels and attach units as described by file - values are in units of erg/s/molecule
    gfactor_00 = df["L00_1e-15"].to_numpy() * 1e-15
    gfactor_10 = df["L10_1e-16"].to_numpy() * 1e-16
    gfactor_11 = df["L11_1e-16"].to_numpy() * 1e-16
    gfactor_22 = df["L22_1e-18"].to_numpy() * 1e-18

    gfactors = gfactor_00 + gfactor_10 + gfactor_11 + gfactor_22

    return OHFluorescenceGFactor1AU(helio_vs, gfactors)


# TODO: change helio_v_kms to be an astropy quantity, add decorator to enforce proper input
# TODO: make rh a parameter and rename function to gfactor: scale the 1AU gfactor data by 1/(rh_in_au)**2
@cache
def hydroxyl_gfactor_1au(helio_v_kms: float) -> float:

    fluorescence_data = read_hydroxyl_gfactor_1au_data()

    # TODO: cite the source of the g factor data
    g1au_interpolation = interp1d(
        fluorescence_data.helio_v_kms,
        fluorescence_data.gfactors_erg_s_molecule,
        kind="cubic",
    )

    # units of ergs/s/molecule
    return g1au_interpolation(helio_v_kms)
