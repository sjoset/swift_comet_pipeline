import numpy as np
import astropy.units as u

from swift_comet_pipeline.types import CometCountRateProfile
from swift_comet_pipeline.types import CometSurfaceBrightnessProfile
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution


# TODO: move this somewhere else
def arcseconds_to_au(arcseconds: float, delta: u.Quantity):
    # TODO: document this and explain magic numbers
    return delta.to(u.AU).value * 2 * np.pi * arcseconds / (3600.0 * 360)  # type: ignore


def countrate_profile_to_surface_brightness(
    countrate_profile: CometCountRateProfile,
    pixel_resolution: SwiftPixelResolution,
    delta: u.Quantity,
) -> CometSurfaceBrightnessProfile:
    """
    CometCountRateProfile holds the count rates, with countrate_profile[x] being the count rate of radius = x pixels
    SwiftPixelResolution holds the number of arcseconds per pixel
    """
    pixel_side_length_cm = (
        arcseconds_to_au(arcseconds=pixel_resolution.value, delta=delta) * u.AU  # type: ignore
    ).to_value(
        u.cm  # type: ignore
    )
    pixel_area_cm2 = pixel_side_length_cm**2

    # surface brightness = count rate per unit area
    surface_brightness_profile = countrate_profile / pixel_area_cm2

    return surface_brightness_profile
