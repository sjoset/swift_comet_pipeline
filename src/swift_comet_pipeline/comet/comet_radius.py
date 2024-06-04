from swift_comet_pipeline.comet.comet_profile import CometProfile
from swift_comet_pipeline.comet.comet_profile_fitting import fit_comet_profile_gaussian
from swift_comet_pipeline.comet.comet_radial_profile import extract_comet_radial_profile
from swift_comet_pipeline.swift.uvot_image import PixelCoord, SwiftUVOTImage


# TODO: re-write to take a pre-generated CometRadialProfile, and possibly the background information,
# and then test when the signal stays at < 1 sigma background for a certain number of pixels:
# maybe a percentage of the radius being considered?  Look ahead the next percent * r pixels and test?
def estimate_comet_radius_at_angle(
    img: SwiftUVOTImage,
    comet_center: PixelCoord,
    radius_guess: int,
    theta: float,
    sigma_threshold: float = 4.0,
) -> float:
    comet_radial_profile = extract_comet_radial_profile(
        img=img,
        comet_center=comet_center,
        theta=theta,
        r=radius_guess,
    )
    comet_profile = CometProfile.from_radial_profile(
        radial_profile=comet_radial_profile
    )

    fitted = fit_comet_profile_gaussian(comet_profile)
    if fitted is None or fitted.stddev is None or fitted.stddev.value is None:
        print("Unable to estimate comet radius! Bad fit?")
        return 0.0

    # go up to a few standard deviations from the center of the comet
    return sigma_threshold * float(fitted.stddev.value)
