import astropy.units as u

from swift_comet_pipeline.types import CometCountRateProfile
from swift_comet_pipeline.types import CometSurfaceBrightnessProfile
from swift_comet_pipeline.types.epoch_summary import EpochSummary


def countrate_profile_to_surface_brightness(
    countrate_profile: CometCountRateProfile,
    epoch_summary: EpochSummary,
) -> CometSurfaceBrightnessProfile:
    """
    Converts count rates to surface brightness based on the physical pixel size
    """

    pixel_side_length = epoch_summary.km_per_pix * u.km  # type: ignore
    pixel_area_cm2 = pixel_side_length.to_value(u.cm) ** 2  # type: ignore

    # surface brightness = count rate per unit area
    surface_brightness_profile = countrate_profile / pixel_area_cm2

    return surface_brightness_profile
