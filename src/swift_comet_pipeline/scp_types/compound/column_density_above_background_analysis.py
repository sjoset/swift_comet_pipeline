from dataclasses import dataclass

import astropy.units as u

from swift_comet_pipeline.scp_types.primitive import *


@dataclass
class ColumnDensityAboveBackgroundAnalysis:
    epoch_id: EpochID
    dust_redness: DustReddeningPercent
    stacking_method: StackingMethod
    last_usable_r: u.Quantity
    last_usable_cd: u.Quantity
    background_oh_cd: u.Quantity
    num_usable_pixels_in_profile: float
    pixel_resolution: UvotPixelResolution
