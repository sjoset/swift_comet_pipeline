from dataclasses import dataclass
from typing import TypeAlias

from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent


@dataclass
class LightCurveEntry:
    observation_time: Time
    time_from_perihelion: u.Quantity

    q: float
    q_err: float

    dust_redness: DustReddeningPercent


LightCurve: TypeAlias = list[LightCurveEntry]
