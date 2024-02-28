from dataclasses import dataclass


@dataclass(frozen=True)
class ValueAndStandardDev:
    value: float
    sigma: float
