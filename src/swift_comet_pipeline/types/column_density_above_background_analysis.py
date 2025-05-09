from dataclasses import dataclass

import astropy.units as u

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution


@dataclass
class ColumnDensityAboveBackgroundAnalysis:
    epoch_id: EpochID
    dust_redness: DustReddeningPercent
    stacking_method: StackingMethod
    last_usable_r: u.Quantity
    last_usable_cd: u.Quantity
    background_oh_cd: u.Quantity
    num_usable_pixels_in_profile: float
    pixel_resolution: SwiftPixelResolution
