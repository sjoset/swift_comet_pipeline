from enum import Enum


# The pixel resolution of the given modes according to the swift documentation
class SwiftPixelResolution(float, Enum):
    # units of arcseconds per pixel
    data_mode = 1.0
    event_mode = 0.502
