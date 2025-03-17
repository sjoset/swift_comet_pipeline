from enum import StrEnum


# maps strings in the image file names to their type
class SwiftUVOTImageType(StrEnum):
    raw = "rw"
    detector = "dt"
    sky_units = "sk"
    exposure_map = "ex"

    @classmethod
    def all_image_types(cls):
        return [x for x in cls]
