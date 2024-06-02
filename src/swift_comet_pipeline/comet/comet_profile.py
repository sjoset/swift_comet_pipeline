from dataclasses import dataclass

import numpy as np

from swift_comet_pipeline.comet.comet_radial_profile import CometRadialProfile


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
