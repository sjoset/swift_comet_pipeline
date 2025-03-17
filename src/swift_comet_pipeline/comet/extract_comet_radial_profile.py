import numpy as np
import pandas as pd
from scipy.integrate import simpson

from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.types import CometRadialProfile
from swift_comet_pipeline.types.count_rate import CountRate, CountRatePerPixel
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def extract_comet_radial_profile(
    img: SwiftUVOTImage, comet_center: PixelCoord, r: int, theta: float
) -> CometRadialProfile:
    """
    Extracts the count rate profile along a line starting at the comet center, extending out a distance r at angle theta
    Takes one pixel sample per unit distance: if r=100, we take 101 pixel samples (to include the center pixel)
    It is important to sample one pixel per r as the processing we do later relies on profile[x] being sampled at radius=x
    """
    # TODO: this should validate that we stay inside the image and truncate the xs and ys to stay inside
    # and return a profile with a smaller r than requested
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


def extract_comet_radial_median_profile_from_cone(
    img: SwiftUVOTImage,
    comet_center: PixelCoord,
    r: int,
    theta: float,
    cone_size: float,
) -> CometRadialProfile:
    """Take a profile of radius r at angle theta, and use profiles from theta +/- cone_size to calculate a median pixel value at each radius"""
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
        extract_comet_radial_profile(
            img=img, comet_center=comet_center, r=r, theta=x
        ).pixel_values
        for x in angles_to_extract
    ]
    median_pixels = np.median(pixel_profiles, axis=0)

    # profile from the middle of the cone, then replace with our calculated median
    middle_radial_profile = extract_comet_radial_profile(
        img=img, comet_center=comet_center, r=r, theta=theta
    )
    middle_radial_profile.pixel_values = median_pixels

    return middle_radial_profile


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
            y=comet_profile.profile_axis_xs * comet_profile.pixel_values,
            x=comet_profile.profile_axis_xs,
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

    return CountRate(value=float(count_rate), sigma=propogated_sigma)


def calculate_distance_from_center_mesh(img: SwiftUVOTImage):
    """
    Resulting array has the same dimensions as the input img, but the pixel values are now the distance to the center of the image,
    rounded to the nearest integer.  This allows addressing a radial profile array with the mesh as the index: radial_profile[distance_from_center_mesh]

    This works if we sample the radial profile at r = 0, r = 1, r = 2, ... but will break if we choose to sample differently
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
    profile: CometRadialProfile,
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


def radial_profile_to_dataframe_product(
    profile: CometRadialProfile, km_per_pix: float
) -> pd.DataFrame:
    """
    Takes a radial profile and returns a dataframe with metadata attached that can be written to disk as a PipelineProduct
    """
    df = pd.DataFrame(
        {
            "r_pixel": profile.profile_axis_xs,
            "count_rate": profile.pixel_values,
            "r_km": profile.profile_axis_xs * km_per_pix,
            "x_pixel": profile._xs,
            "y_pixel": profile._ys,
        }
    )
    df.attrs.update(
        {
            "radius": profile._radius,
            "theta": profile._theta,
            "comet_center_x": profile._comet_center.x,
            "comet_center_y": profile._comet_center.y,
        }
    )
    return df


def radial_profile_from_dataframe_product(df: pd.DataFrame) -> CometRadialProfile:
    """
    Reads a saved PipelineProduct we stored with radial_profile_from_dataframe_product and reconstructs a radial profile
    """

    return CometRadialProfile(
        profile_axis_xs=df.r_pixel.values,
        pixel_values=df.count_rate.values,
        _xs=df.x_pixel.values,
        _ys=df.y_pixel.values,
        _radius=df.attrs["radius"],
        _theta=df.attrs["theta"],
        _comet_center=PixelCoord(
            x=df.attrs["comet_center_x"], y=df.attrs["comet_center_y"]
        ),
    )
