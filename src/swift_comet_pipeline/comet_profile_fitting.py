from typing import Optional
import numpy as np
from astropy import modeling
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt

from swift_comet_pipeline.comet_profile import (
    CometProfile,
    extract_comet_radial_profile,
)
from swift_comet_pipeline.uvot_image import PixelCoord, SwiftUVOTImage


__all__ = [
    "fit_comet_profile_gaussian",
    "plot_fitted_gaussian_profile",
    "estimate_comet_radius_at_angle",
]


def fit_comet_profile_gaussian(comet_profile: CometProfile) -> Optional[Gaussian1D]:
    """Takes a CometRadialProfile and returns a function f(r) generated from the profile's best-fit gaussian"""

    if not comet_profile.center_is_comet_peak:
        print("This function is intended for profiles centered on the comet peak!")
        return None

    lsqf = modeling.fitting.LevMarLSQFitter()
    gaussian = modeling.models.Gaussian1D()
    fitted_gaussian = lsqf(
        gaussian, comet_profile.profile_axis_xs, comet_profile.pixel_values
    )

    return fitted_gaussian


def plot_fitted_gaussian_profile(
    comet_profile: CometProfile,
    fitted_model: Gaussian1D,
    sigma_threshold: float,
    plot_title: str,
) -> None:
    if fitted_model.stddev.value is None:
        print("Fitted gaussian has no information about standard deviation!")
    else:
        plt.vlines(
            x=[
                -sigma_threshold * fitted_model.stddev.value,
                sigma_threshold * fitted_model.stddev.value,
            ],
            ymin=0,
            ymax=np.max(comet_profile.pixel_values),
            color="r",
        )
    plt.scatter(comet_profile.profile_axis_xs, comet_profile.pixel_values)
    plt.plot(
        comet_profile.profile_axis_xs,
        fitted_model(comet_profile.profile_axis_xs),
    )
    plt.title(plot_title)
    plt.show()


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


# def fit_inverse_r(img: SwiftUVOTImage) -> None:
#     profile_radius = 40
#
#     pix_center = get_uvot_image_center(img=img)
#     search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
#     peak = find_comet_center(
#         img=img,
#         method=CometCenterFindingMethod.aperture_peak,
#         search_aperture=search_aperture,
#     )
#     comet_profile = count_rate_profile(
#         img=img,
#         comet_center=peak,
#         theta=0,
#         r=profile_radius,
#     )
#
#     mask = comet_profile.distances_from_center > 0
#     rs = np.log10(comet_profile.distances_from_center[mask])
#     pix = comet_profile.pixel_values[mask]
#
#     def log_dust_profile(r, a, b):
#         return a * r + b
#
#     dust_fit = curve_fit(
#         log_dust_profile,
#         rs,
#         pix,
#         [-1, 0],
#     )
#
#     a_fit = dust_fit[0][0]
#     b_fit = dust_fit[0][1]
#
#     print(f"{a_fit=}, {b_fit=}")
#
#     plt.plot(
#         rs,
#         log_dust_profile(rs, a_fit, b_fit),
#     )
#     plt.plot(
#         np.log10(np.abs(comet_profile.distances_from_center)),
#         comet_profile.pixel_values,
#     )
#     plt.show()
