from dataclasses import dataclass


@dataclass
class PixelCoord:
    """Use floats instead of ints to allow sub-pixel addressing if we need"""

    x: float
    y: float
