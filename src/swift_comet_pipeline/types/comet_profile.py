from dataclasses import dataclass

import numpy as np

from swift_comet_pipeline.types.pixel_coord import PixelCoord


@dataclass
class CometRadialProfile:
    """
    Count rate values along a line extending from the comet center out to radius r at angle theta
    Theta is measured from the positive x-axis (along a numpy row) counter-clockwise
    One sample is taken of the profile per unit radial distance: If we want a cut of radius 20, we will have 20
    (x, y) pairs sampled.  We will also have the comet center at radius zero for a total of 21 points in the resulting profile.

    The stacking step adjusts the image pixels to be in count rate, so pixel_values will be an array of floats representing count rates
    """

    # TODO: rename profile_axis_xs or alias it to 'r'

    # the distance from comet center of each sample along the line in pixels - these are x coordinates along the profile axis, with pixel_values being the y values
    # these are not simply [r=0, r=1, r=2, ...] but calculated from the x, y coordinates of the pixels involved
    profile_axis_xs: np.ndarray
    # the actual pixel values (count rates)
    pixel_values: np.ndarray

    # the (x, y) pixel coordinates of each pixel sample along the profile
    _xs: np.ndarray
    _ys: np.ndarray
    # The angle at which we cut, measured counter-clockwise from the positive x axis (to the right - along a row of the image),
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
