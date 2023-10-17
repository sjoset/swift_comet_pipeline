from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import Gaussian1D
from scipy.integrate import simpson
from astropy import modeling

from uvot_image import PixelCoord, SwiftUVOTImage
from count_rate import CountRate, CountRatePerPixel


__all__ = [
    "CometRadialProfile",
    "CometProfile",
    "extract_comet_radial_profile",
    "count_rate_from_comet_radial_profile",
    "count_rate_from_comet_profile",
    "fit_comet_profile_gaussian",
    "plot_fitted_profile",
    "estimate_comet_radius_at_angle",
]


@dataclass
class CometRadialProfile:
    """
    Count rate values along a line extending from the comet center out to radius r at angle theta
    One sample is taken of the profile per unit radial distance: If we want a cut of radius 20, we will have 20
    (x, y) pairs sampled.  We will also have the comet center at radius zero for a total of 21 points in the resulting profile.
    """

    # the distance from comet center of each sample along the line - these are x coordinates along the profile axis, with pixel_values being the y values
    profile_axis_xs: np.ndarray
    # the actual pixel values (count rates)
    pixel_values: np.ndarray

    # the (x, y) pixel coordinates of each pixel sample along the profile
    _xs: np.ndarray
    _ys: np.ndarray
    # The angle at which we cut, measured from the positive x axis (to the right along a row of the image),
    # and how far this profile cut extends
    _radius: int
    _theta: float
    # coordinates used for the center of the comet, in case we need those later
    _comet_center: PixelCoord


@dataclass
class CometProfile:
    """CometRadialProfile can be assumed to extend outward from the center of the comet, while this structure is intended for any arbitrary profile of pixel values"""

    # as in CometRadialProfile
    profile_axis_xs: np.ndarray
    pixel_values: np.ndarray

    # keep track of whether this is an arbitrary profile, or a special comet-centered profile
    center_is_comet_peak: bool

    # the (x, y) pixel coordinates of each pixel sample along the profile
    _xs: np.ndarray
    _ys: np.ndarray

    @classmethod
    def from_radial_profile(cls, radial_profile: CometRadialProfile):
        """Mirror the given radial profile about the center of the comet, simulating slice of size r along theta with slice of size r along -theta"""
        x0 = radial_profile._comet_center.x
        y0 = radial_profile._comet_center.y
        r = radial_profile._radius
        theta = radial_profile._theta

        x1 = x0 - r * np.cos(theta)
        y1 = y0 - r * np.sin(theta)

        # we have the pixel in the center, plus r pixels in the direction away from the center
        num_samples = r + 1

        # x, y coordinate of pixel sampling, starting closest from the center and moving outward
        xs = np.linspace(np.round(x0), np.round(x1), num=num_samples, endpoint=True)
        ys = np.linspace(np.round(y0), np.round(y1), num=num_samples, endpoint=True)

        # flip xs, ys around so that they start at most distant point from center at the beginning of the array,
        # removing the pixel at the comet center as that is already included in radial_profile
        xs = np.array(list(reversed(xs[1:])))
        ys = np.array(list(reversed(ys[1:])))

        # mirror the pixel values around the center just like we did with the xs and ys
        pixel_values = np.array(list(reversed(radial_profile.pixel_values[1:])))

        # and the x-coordinates along the profile need to be flipped as well
        profile_axis_xs = -np.array(list(reversed(radial_profile.profile_axis_xs[1:])))

        return CometProfile(
            profile_axis_xs=np.concatenate(
                (profile_axis_xs, radial_profile.profile_axis_xs), axis=None
            ),
            pixel_values=np.concatenate(
                (pixel_values, radial_profile.pixel_values), axis=None
            ),
            center_is_comet_peak=True,
            _xs=np.concatenate((xs, radial_profile._xs), axis=None),
            _ys=np.concatenate((ys, radial_profile._ys), axis=None),
        )


def extract_comet_radial_profile(
    img: SwiftUVOTImage, comet_center: PixelCoord, r: int, theta: float
) -> CometRadialProfile:
    """Extracts the count rate profile along a line starting at the comet center, extending out a distance r at angle theta"""
    x0 = comet_center.x
    y0 = comet_center.y
    x1 = comet_center.x + r * np.cos(theta)
    y1 = comet_center.y + r * np.sin(theta)

    # we have the pixel in the center, plus r pixels in the direction away from the center
    num_samples = r + 1

    xs = np.linspace(np.round(x0), np.round(x1), num=num_samples, endpoint=True)
    ys = np.linspace(np.round(y0), np.round(y1), num=num_samples, endpoint=True)

    pixel_values = img[ys.astype(np.int32), xs.astype(np.int32)]

    distances_from_center = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)

    return CometRadialProfile(
        profile_axis_xs=distances_from_center,
        pixel_values=pixel_values,
        _xs=xs,
        _ys=ys,
        _radius=r,
        _theta=theta,
        _comet_center=comet_center,
    )


def count_rate_from_comet_radial_profile(
    comet_profile: CometRadialProfile,
    bg: CountRatePerPixel,
) -> CountRate:
    """
    Takes a radial profile and assumes azimuthal symmetry to produce a count rate that would result
    from a circular aperture centered on the comet profile
    Reminder: we need the background count rate to propogate error
    """

    # our integral is (count rate at r) * (r dr dtheta) for the total count rate

    count_rate = (
        simpson(
            comet_profile.profile_axis_xs * comet_profile.pixel_values,
            comet_profile.profile_axis_xs,
        )
        * 2
        * np.pi
    )

    # uncertainty of this integral
    profile_sigma = np.std(comet_profile.pixel_values) * 2 * np.pi

    # quadrature for our error plus the error accumulated from the background over the area of the comet
    propogated_sigma = np.sqrt(
        profile_sigma**2 + (np.pi * comet_profile._radius**2 * bg.sigma**2)
    )

    return CountRate(value=count_rate, sigma=propogated_sigma)


def count_rate_from_comet_profile(
    comet_profile: CometProfile,
    bg: CountRatePerPixel,
) -> Optional[CountRate]:
    """
    Takes a profile and assumes azimuthal symmetry to produce a count rate that would result
    from a circular aperture centered on the middle of the comet profile
    Reminder: we need the background count rate to propogate error
    """

    if not comet_profile.center_is_comet_peak:
        print("This function is intended for profiles centered on the comet peak!")
        return None

    # The factor is normally 2 * np.pi for the angular part of the integral, but
    # our answer is twice as big because our limits on r run from [-r, ..., r] and not [0, ..., r].

    count_rate = (
        simpson(
            np.abs(comet_profile.profile_axis_xs) * comet_profile.pixel_values,
            comet_profile.profile_axis_xs,
        )
        * np.pi
    )

    # TODO: find how to properly quantify the error for these pixel values
    profile_sigma = np.std(comet_profile.pixel_values) * 2 * np.pi

    # pull radius of the aperture from the max distance in the profile
    propogated_sigma = np.sqrt(
        profile_sigma**2
        + (np.pi * np.max(comet_profile.profile_axis_xs) ** 2 * bg.sigma**2)
    )

    return CountRate(value=count_rate, sigma=propogated_sigma)


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


def plot_fitted_profile(
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
