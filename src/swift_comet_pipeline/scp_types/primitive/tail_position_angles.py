from dataclasses import dataclass

import astropy.units as u


@dataclass
class TailPositionAngles:
    dust_tail_pa: u.Quantity
    ion_tail_pa: u.Quantity
