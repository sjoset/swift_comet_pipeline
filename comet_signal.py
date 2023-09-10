import numpy as np
from scipy.integrate import simpson
from photutils.aperture import CircularAperture, ApertureStats
from astropy import modeling

import matplotlib.pyplot as plt

from typing import Optional, Tuple, Callable
from enum import StrEnum, auto
from dataclasses import dataclass

from uvot_image import PixelCoord, SwiftUVOTImage, get_uvot_image_center
from count_rate import CountRate, CountRatePerPixel


__all__ = [
    "CometCenterFindingMethod",
    "CometProfile",
    "comet_manual_aperture",
    "find_comet_center",
    "count_rate_profile",
    "count_rate_from_count_rate_profile",
    "fit_comet_profile_gaussian",
    "estimate_comet_radius_by_angle",
    "estimate_comet_radius_at_angle",
]


class CometCenterFindingMethod(StrEnum):
    pixel_center = auto()
    aperture_centroid = auto()
    aperture_peak = auto()

    @classmethod
    def all_methods(cls):
        return [x for x in cls]


def comet_manual_aperture(
    img: SwiftUVOTImage,
    aperture_x: float,
    aperture_y: float,
    aperture_radius: float,
    bg: CountRatePerPixel,
) -> CountRate:
    comet_aperture = CircularAperture((aperture_x, aperture_y), r=aperture_radius)
    comet_aperture_stats = ApertureStats(img, comet_aperture)

    comet_count_rate = float(comet_aperture_stats.sum)
    # TODO: this is not a good error calculation but it will have to do for now
    comet_count_rate_sigma = comet_aperture_stats.std

    propogated_sigma = np.sqrt(
        comet_count_rate_sigma**2
        + (comet_aperture_stats.sum_aper_area.value * bg.sigma**2)
    )

    return CountRate(value=comet_count_rate, sigma=propogated_sigma)


def find_comet_center(
    img: SwiftUVOTImage,
    method: CometCenterFindingMethod,
    search_aperture: Optional[CircularAperture] = None,
) -> PixelCoord:
    """
    Coordinates returned are x, y values
    """
    if method == CometCenterFindingMethod.pixel_center:
        return get_uvot_image_center(img=img)
    elif method == CometCenterFindingMethod.aperture_centroid:
        return comet_center_by_centroid(img=img, search_aperture=search_aperture)
    elif method == CometCenterFindingMethod.aperture_peak:
        return comet_center_by_peak(img=img, search_aperture=search_aperture)


def comet_center_by_centroid(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> PixelCoord:
    if search_aperture is None:
        print("No aperture provided for center finding by centroid!")
        return PixelCoord(-1.0, -1.0)

    stats = ApertureStats(img, search_aperture)

    # return tuple(stats.centroid)
    return PixelCoord(x=stats.centroid[0], y=stats.centroid[1])


def comet_center_by_peak(
    img: SwiftUVOTImage, search_aperture: Optional[CircularAperture]
) -> PixelCoord:
    if search_aperture is None:
        print("No aperture provided for center finding by peak!")
        return PixelCoord(-1.0, -1.0)

    # cut out the pixels in the aperture
    ap_mask = search_aperture.to_mask(method="center")
    img_cutout = ap_mask.cutout(data=img)  # type: ignore

    # index of peak value for img represented as 1d list
    peak_pos_raveled = np.argmax(img_cutout)
    # unravel turns this 1d index into (row, col) indices
    peak_pos = np.unravel_index(peak_pos_raveled, img_cutout.shape)

    ap_min_x, ap_min_y = search_aperture.bbox.ixmin, search_aperture.bbox.iymin  # type: ignore

    # return (float(ap_min_x + peak_pos[1]), float(ap_min_y + peak_pos[0]))
    return PixelCoord(x=ap_min_x + peak_pos[1], y=ap_min_y + peak_pos[0])


@dataclass
class CometProfile:
    xs: np.ndarray
    ys: np.ndarray
    distances_from_center: np.ndarray
    pixel_values: np.ndarray
    radius: float
    theta: float


def count_rate_profile(
    img: SwiftUVOTImage, comet_center: PixelCoord, r: int, theta: float
) -> CometProfile:
    x0 = comet_center.x - r * np.cos(theta)
    y0 = comet_center.y - r * np.sin(theta)
    x1 = comet_center.x + r * np.cos(theta)
    y1 = comet_center.y + r * np.sin(theta)

    # we have the pixel in the center, plus r pixels in each direction away from the center
    num_samples = 2 * r + 1

    xs = np.linspace(np.round(x0), np.round(x1), num_samples)
    ys = np.linspace(np.round(y0), np.round(y1), num_samples)

    pixel_values = img[ys.astype(np.int32), xs.astype(np.int32)]

    distances_from_center = np.array(range(-r, r + 1))

    return CometProfile(
        xs=xs,
        ys=ys,
        distances_from_center=distances_from_center,
        pixel_values=pixel_values,
        radius=r,
        theta=theta,
    )


def count_rate_from_count_rate_profile(
    comet_profile: CometProfile,
    bg: CountRatePerPixel,
) -> CountRate:
    """
    Takes a profile and assumes azimuthal symmetry to produce a count rate that would result
    from a circular aperture centered on the middle of the comet profile
    """

    # our radii run negative to positive, so we need absolute values here
    # The factor is normally 2 * np.pi for the angular part of the integral, but
    # our answer is twice as big because our limits on r run from [-r, ..., r] and not [0, ..., r].

    count_rate = (
        simpson(
            np.abs(comet_profile.distances_from_center) * comet_profile.pixel_values,
            comet_profile.distances_from_center,
        )
        * np.pi
    )

    profile_sigma = np.std(comet_profile.pixel_values) * np.pi

    # pull radius of the aperture from the max distance in the profile
    propogated_sigma = np.sqrt(
        profile_sigma**2
        + (np.pi * np.max(comet_profile.distances_from_center) ** 2 * bg.sigma**2)
    )

    return CountRate(value=count_rate, sigma=propogated_sigma)


def fit_comet_profile_gaussian(comet_profile: CometProfile) -> Callable:
    lsqf = modeling.fitting.LevMarLSQFitter()
    gaussian = modeling.models.Gaussian1D()
    fitted_gaussian = lsqf(
        gaussian, comet_profile.distances_from_center, comet_profile.pixel_values
    )

    return fitted_gaussian


def estimate_comet_radius_by_angle(
    img: SwiftUVOTImage,
    comet_center: PixelCoord,
    radius_guess: int,
    num_profile_slices: int = 50,
    sigma_threshold: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    thetas = np.linspace(0, np.pi, num_profile_slices, endpoint=False)
    comet_profiles = [
        count_rate_profile(img=img, comet_center=comet_center, theta=x, r=radius_guess)
        for x in thetas
    ]

    profile_fits = [fit_comet_profile_gaussian(profile) for profile in comet_profiles]

    radii = np.array([sigma_threshold * fit.stddev.value for fit in profile_fits])

    return thetas, radii


def estimate_comet_radius_at_angle(
    img: SwiftUVOTImage,
    comet_center: PixelCoord,
    radius_guess: int,
    theta: float,
    sigma_threshold: float = 4.0,
) -> float:
    comet_profile = count_rate_profile(
        img=img,
        comet_center=comet_center,
        theta=theta,
        r=radius_guess,
    )

    fitted = fit_comet_profile_gaussian(comet_profile)

    # go up to a few standard deviations from the center of the comet
    return sigma_threshold * fitted.stddev.value


def plot_fitted_profile(
    comet_profile: CometProfile, fitted_model, sigma_threshold: float, plot_title: str
) -> None:
    plt.vlines(
        x=[
            -sigma_threshold * fitted_model.stddev.value,
            sigma_threshold * fitted_model.stddev.value,
        ],
        ymin=0,
        ymax=np.max(comet_profile.pixel_values),
        color="r",
    )
    plt.scatter(comet_profile.distances_from_center, comet_profile.pixel_values)
    plt.plot(
        comet_profile.distances_from_center,
        fitted_model(comet_profile.distances_from_center),
    )
    plt.title(plot_title)
    plt.show()
