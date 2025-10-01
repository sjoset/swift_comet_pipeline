from dataclasses import dataclass

from astropy.time import Time


@dataclass
class OrbitPerihelion:
    t_perihelion: Time
    rh_au: float
