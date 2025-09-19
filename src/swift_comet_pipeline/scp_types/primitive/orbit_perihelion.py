from dataclasses import dataclass

from astropy.time import Time
import astropy.units as u


@dataclass
class OrbitPerihelion:
    t_perihelion: Time
    r_h: u.Quantity
