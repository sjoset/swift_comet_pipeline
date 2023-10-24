from dataclasses import dataclass


@dataclass
class ValueAndStandardDev:
    value: float
    sigma: float
