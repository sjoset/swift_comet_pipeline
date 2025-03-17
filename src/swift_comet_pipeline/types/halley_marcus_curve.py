from dataclasses import dataclass
from typing import TypeAlias

# __all__ = ["HalleyMarcusCurveEntry", "HalleyMarcusCurve"]


@dataclass
class HalleyMarcusCurveEntry:
    phase_deg: float

    # correction factor when normalizing to 0 degree phase
    phase_correction_zero: float
    # correction factor when normalizing to 90 degree phase
    phase_correction_ninety: float


HalleyMarcusCurve: TypeAlias = list[HalleyMarcusCurveEntry]
