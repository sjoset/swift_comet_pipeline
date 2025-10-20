import astropy.units as u


from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry


def oh_countrate_profile_to_surface_brightness(
    oh_countrate_profile: CometCountRateProfile,
    eid: EpochIndexEntry,
) -> CometSurfaceBrightnessProfile:
    """
    Converts count rates to surface brightness based on the physical pixel size
    """

    pixel_side_length = eid.km_per_pix * u.km  # type: ignore
    pixel_area_cm2 = pixel_side_length.to_value(u.cm) ** 2  # type: ignore

    # surface brightness = count rate per unit area
    surface_brightness_profile = oh_countrate_profile / pixel_area_cm2

    return surface_brightness_profile
