from astropy.coordinates import Angle
import numpy as np
from scipy.integrate import simpson

from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.scp_types.compound.background_result import BackgroundResult
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.swift.swift_orientation import swift_position_angle


def extract_comet_radial_profile_from_cone(
    img: SwiftUvotImage, comet_center: PixelCoord, r: int, theta: float
) -> CometRadialProfileFromConicalRegion:
    """
    Extracts the count rate profile along a line starting at the comet center, extending out a distance r at angle theta
    Takes one pixel sample per unit distance: if r=100, we take 101 pixel samples (to include the center pixel)
    It is important to sample one pixel per r as the processing we do later relies on profile[x] being sampled at radius=x
    """
    # TODO: we could use the exposure map to enforce only maximally-exposed pixels for our profiles
    # TODO: this should validate that we stay inside the image and truncate the xs and ys to stay inside
    # and return a profile with a smaller r than requested
    x0 = comet_center.x
    y0 = comet_center.y
    x1 = comet_center.x + r * np.cos(theta)
    y1 = comet_center.y + r * np.sin(theta)

    # we have the pixel in the center, plus r pixels in the direction away from the center
    num_samples = r + 1

    # TODO: if round(x0) == round(x1) this fails to produce a linspace!

    xs = np.linspace(np.round(x0), np.round(x1), num=num_samples, endpoint=True)
    ys = np.linspace(np.round(y0), np.round(y1), num=num_samples, endpoint=True)

    pixel_values = img[ys.astype(np.int32), xs.astype(np.int32)]

    distances_from_center = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)
    pa = swift_position_angle(Angle(theta * u.deg))  # type: ignore

    return CometRadialProfileFromConicalRegion(
        profile_axis_rs=distances_from_center,
        pixel_values=pixel_values,
        _xs=xs,
        _ys=ys,
        _radius=r,
        _theta=theta,
        _position_angle=pa.to_value(u.deg),  # type: ignore
        _comet_center=comet_center,
    )


def extract_comet_radial_median_profile_from_cone(
    img: SwiftUvotImage,
    comet_center: PixelCoord,
    r: int,
    theta: float,
    cone_size: float,
) -> CometRadialProfileFromConicalRegion:
    """
    Take a profile of radius r at angle theta, and use profiles from theta +/- cone_size to calculate a median pixel value at each radius

    Angles are given in radians, cone_size is angular size in radians from the center, so that the whole angular cone size is 2*cone_size
    """
    extraction_cone_mid_angle = theta
    extraction_cone_min_angle = extraction_cone_mid_angle - cone_size
    extraction_cone_max_angle = extraction_cone_mid_angle + cone_size

    # extract a profile for every pixel at the edge of the cone
    cone_arclength_pixels = int(np.abs(np.round(2 * theta * r)))
    angles_to_extract = np.linspace(
        extraction_cone_min_angle, extraction_cone_max_angle, cone_arclength_pixels
    )

    # take the median value at each radius
    pixel_profiles = [
        extract_comet_radial_profile_from_cone(
            img=img, comet_center=comet_center, r=r, theta=x
        ).pixel_values
        for x in angles_to_extract
    ]
    median_pixels = np.median(pixel_profiles, axis=0)

    # profile from the middle of the cone, then replace with our calculated median
    middle_radial_profile = extract_comet_radial_profile_from_cone(
        img=img, comet_center=comet_center, r=r, theta=theta
    )
    middle_radial_profile.pixel_values = median_pixels
    middle_radial_profile._cone_size = cone_size

    return middle_radial_profile


def total_count_rate_from_comet_radial_profile(
    comet_profile: CometRadialProfileFromConicalRegion,
    bgr: BackgroundResult,
    t_exp_s: float,
) -> CountRate:
    """
    Takes a radial profile and assumes azimuthal symmetry to produce a count rate that would result
    from a circular aperture centered on the comet profile
    """

    # our integral is (count rate at r) * (r dr dtheta) for the total count rate
    count_rate = (
        simpson(
            y=comet_profile.profile_axis_rs * comet_profile.pixel_values,
            x=comet_profile.profile_axis_rs,
        )
        * 2
        * np.pi
    )

    comet_area = np.pi * comet_profile._radius**2
    bg_area = bgr.bg_num_pixels

    profile_variance = np.sum(comet_profile.pixel_values - bgr.b_hat) / t_exp_s

    profile_variance += (
        comet_area
        * bgr.bg_shot_noise_variance**2
        * (1 + (np.pi / 2) * (comet_area / bg_area))
    )

    return CountRate(value=float(count_rate), sigma=np.sqrt(profile_variance))


def calculate_distance_from_center_mesh(img: SwiftUvotImage):
    """
    Resulting array has the same dimensions as the input img, but the pixel values are now the distance to the center of the image,
    rounded to the nearest integer.  This allows addressing a radial profile array with the mesh as the index: radial_profile[distance_from_center_mesh]

    This works if we sample the radial profile at r = 0, r = 1, r = 2, ... but will break if we choose to sample differently

    TODO: can we replace this with
    Y, X = np.ogrid
    dist = np.hypot((X - center_x), (Y - center_y))?
    """

    img_height, img_width = img.shape
    img_center = get_uvot_image_center(img=img)
    xs = np.linspace(0, img_width, num=img_width, endpoint=False)
    ys = np.linspace(0, img_height, num=img_height, endpoint=False)
    x, y = np.meshgrid(xs, ys)

    # the pixel values in the mesh image are the distance from the center, rounded to the nearest integer, so we can use
    # these values as an index to create a radially symmetric image from a 1-dimensional profile
    distance_from_center_mesh = np.round(
        np.sqrt((x - img_center.x) ** 2 + (y - img_center.y) ** 2)
    ).astype(int)

    return distance_from_center_mesh


def radial_profile_to_image(
    profile: CometRadialProfileFromConicalRegion,
    distance_from_center_mesh: np.ndarray,
    empty_pixel_fill_value: float = 0.0,
):
    """
    The array distance_from_center_mesh is assumed to be a 2d array, whose values express the distance from that pixel to the center of the image, rounded to the nearest integer.
    To generate the image easily, we zero-pad the comet's radial profile out to the maximum distance specified in this mesh (if necessary)
    """
    max_dist = np.max(distance_from_center_mesh)

    num_extra_pixels = max_dist - len(profile.pixel_values) + 1
    if num_extra_pixels >= 1:
        extended_profile = np.pad(
            profile.pixel_values,
            (0, num_extra_pixels),
            mode="constant",
            constant_values=(empty_pixel_fill_value, empty_pixel_fill_value),
        )
    else:
        extended_profile = profile.pixel_values
    img = extended_profile[distance_from_center_mesh]
    return img
