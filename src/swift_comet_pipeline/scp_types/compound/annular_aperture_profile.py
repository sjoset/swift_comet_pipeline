from dataclasses import dataclass, asdict
from typing import TypeAlias

import numpy as np
import pandas as pd

from swift_comet_pipeline.scp_types.compound.comet_profile import CometRadialProfile
from swift_comet_pipeline.scp_types.primitive.aperture_count_rate_analysis import (
    ApertureCountRateAnalysis,
)


@dataclass
class AnnularApertureProfileEntry(ApertureCountRateAnalysis):
    """
    For storing pairs of annular aperture radius and width (dr)
    """

    aperture_r_pix: float
    aperture_r_km: float

    aperture_dr_pix: float
    aperture_dr_km: float


AnnularApertureProfile: TypeAlias = list[AnnularApertureProfileEntry]


def dataframe_from_annular_aperture_profile(
    annular_aperture_profile: AnnularApertureProfile,
) -> pd.DataFrame:
    return pd.DataFrame(data=[asdict(aape) for aape in annular_aperture_profile])


def radial_profile_from_annular_aperture_profile(
    annular_aperture_profile: AnnularApertureProfile,
) -> CometRadialProfile:

    profile_rs_pix = np.array([x.aperture_r_pix for x in annular_aperture_profile])
    pixel_values = np.array([x.median_count_rate for x in annular_aperture_profile])

    return CometRadialProfile(profile_axis_rs=profile_rs_pix, pixel_values=pixel_values)
