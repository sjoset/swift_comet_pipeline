from enum import Enum


# Maps the strings in FITS file header under the keyword DATAMODE
class SwiftImageMode(str, Enum):
    data_mode = "IMAGE"
    event_mode = "EVENT"
