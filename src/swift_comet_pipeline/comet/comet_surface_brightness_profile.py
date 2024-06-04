from typing import TypeAlias

import numpy as np
import astropy.units as u

from swift_comet_pipeline.comet.comet_count_rate_profile import CometCountRateProfile
from swift_comet_pipeline.swift.uvot_image import SwiftPixelResolution


CometSurfaceBrightnessProfile: TypeAlias = np.ndarray


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


# TODO: move this surface brightness to another file
# TODO: finish this dataclass to reflect the dataframe returned by surface_brightness_profiles()
# @dataclass
# class SurfaceBrightnessProfiles:
#     r_inner_pix: np.ndarray
#     r_outer_pix: np.ndarray


# # TODO: deprecate?
# def surface_brightness_profiles(
#     uw1: SwiftUVOTImage, uvv: SwiftUVOTImage, r_max: int
# ) -> pd.DataFrame:
#     # fix the annular aperture thickness at one pixel
#     radial_slice_thickness: int = 1
#
#     comet_peak = get_uvot_image_center(uw1)
#     # first slice is a circular aperture centered on the comet, after which we move to annular apertures
#     center_aperture = CircularAperture(
#         positions=(comet_peak.x, comet_peak.y), r=radial_slice_thickness
#     )
#     center_uw1_stats = ApertureStats(data=uw1, aperture=center_aperture)
#     center_uvv_stats = ApertureStats(data=uvv, aperture=center_aperture)
#
#     # construct annular apertures
#     r_inner_max = r_max - radial_slice_thickness
#     num_apertures = r_max - radial_slice_thickness
#     inner_aperture_rs = np.linspace(
#         start=radial_slice_thickness, stop=r_inner_max, endpoint=True, num=num_apertures
#     )
#     annular_apertures = [
#         CircularAnnulus((comet_peak.x, comet_peak.y), r_in=r_inner, r_out=r_inner + 1)
#         for r_inner in inner_aperture_rs
#     ]
#
#     uw1_stats = [ApertureStats(data=uw1, aperture=ap) for ap in annular_apertures]
#     uvv_stats = [ApertureStats(data=uvv, aperture=ap) for ap in annular_apertures]
#
#     # add results to the dataframe, by pasting together the center results with the results from
#     # the aperture list
#     df = pd.DataFrame()
#     df["r_inner_pix"] = np.append(np.array([0]), inner_aperture_rs)
#     df["r_outer_pix"] = df["r_inner_pix"] + 1
#     df["surface_brightness_uw1_median"] = np.append(
#         center_uw1_stats.median, [x.median for x in uw1_stats]
#     )
#     df["surface_brightness_uw1_median_err"] = np.append(
#         center_uw1_stats.std, [1.2533 * x.std for x in uw1_stats]
#     )
#     df["surface_brightness_uw1_mean"] = np.append(
#         center_uw1_stats.mean, [x.mean for x in uw1_stats]
#     )
#     df["surface_brightness_uw1_mean_err"] = np.append(
#         center_uw1_stats.std, [x.std for x in uw1_stats]
#     )
#     df["surface_brightness_uvv_median"] = np.append(
#         center_uvv_stats.median, [x.median for x in uvv_stats]
#     )
#     df["surface_brightness_uvv_median_err"] = np.append(
#         center_uvv_stats.std, [1.2533 * x.std for x in uvv_stats]
#     )
#     df["surface_brightness_uvv_mean"] = np.append(
#         center_uvv_stats.mean, [x.mean for x in uvv_stats]
#     )
#     df["surface_brightness_uvv_mean_err"] = np.append(
#         center_uvv_stats.std, [x.std for x in uvv_stats]
#     )
#
#     df["cumulative_counts_uw1_mean"] = df.surface_brightness_uw1_mean.cumsum()
#     df["cumulative_counts_uw1_mean_err"] = np.sqrt(
#         np.cumsum(np.square(df.cumulative_counts_uw1_mean))
#     )
#     df["cumulative_counts_uw1_median"] = df.surface_brightness_uw1_median.cumsum()
#     df["cumulative_counts_uw1_median_err"] = np.sqrt(
#         np.cumsum(np.square(df.cumulative_counts_uw1_median))
#     )
#     df["cumulative_counts_uvv_mean"] = df.surface_brightness_uvv_mean.cumsum()
#     df["cumulative_counts_uvv_mean_err"] = np.sqrt(
#         np.cumsum(np.square(df.cumulative_counts_uvv_mean))
#     )
#     df["cumulative_counts_uvv_median"] = df.surface_brightness_uvv_median.cumsum()
#     df["cumulative_counts_uvv_median_err"] = np.sqrt(
#         np.cumsum(np.square(df.cumulative_counts_uvv_median))
#     )
#
#     return df


# TODO: deprecate?
# def qh2o_from_surface_brightness_profiles(
#     df: pd.DataFrame, epoch: Epoch, beta: float
# ) -> pd.DataFrame:
#     helio_r_au = np.mean(epoch.HELIO)
#     helio_v_kms = np.mean(epoch.HELIO_V)
#     delta = np.mean(epoch.OBS_DIS)
#
#     df["oh_brightness_median"] = (
#         df.surface_brightness_uw1_median - beta * df.surface_brightness_uvv_median
#     )
#     df["oh_brightness_mean"] = (
#         df.surface_brightness_uw1_mean - beta * df.surface_brightness_uvv_mean
#     )
#     df["oh_brightness_running_total_median"] = df.oh_brightness_median.cumsum()
#     df["oh_brightness_running_total_mean"] = df.oh_brightness_mean.cumsum()
#
#     flux_median = []
#     flux_median_err = []
#     flux_median_max = []
#     flux_median_max_err = []
#     for _, row in df.iterrows():
#         uw1cr = CountRate(
#             value=row.cumulative_counts_uw1_median,
#             sigma=row.cumulative_counts_uw1_median_err,
#         )
#         uvvcr = CountRate(
#             value=row.cumulative_counts_uvv_median,
#             sigma=row.cumulative_counts_uvv_median_err,
#         )
#         flux_OH = OH_flux_from_count_rate(uw1=uw1cr, uvv=uvvcr, beta=beta)
#         flux_median.append(flux_OH.value)
#         flux_median_err.append(flux_OH.sigma)
#
#         flux_max = OH_flux_from_count_rate(
#             uw1=uw1cr, uvv=CountRate(value=0.0, sigma=0.0), beta=beta
#         )
#         flux_median_max.append(flux_max.value)
#         flux_median_max_err.append(flux_max.sigma)
#
#     df["flux_median"] = flux_median
#     df["flux_median_err"] = flux_median_err
#     df["flux_median_max"] = flux_median_max
#     df["flux_median_max_err"] = flux_median_max_err
#
#     flux_mean = []
#     flux_mean_err = []
#     for _, row in df.iterrows():
#         uw1cr = CountRate(
#             value=row.cumulative_counts_uw1_mean,
#             sigma=row.cumulative_counts_uw1_mean_err,
#         )
#         uvvcr = CountRate(
#             value=row.cumulative_counts_uvv_mean,
#             sigma=row.cumulative_counts_uvv_mean_err,
#         )
#         flux_OH = OH_flux_from_count_rate(uw1=uw1cr, uvv=uvvcr, beta=beta)
#         flux_mean.append(flux_OH.value)
#         flux_mean_err.append(flux_OH.sigma)
#
#     df["flux_mean"] = flux_mean
#     df["flux_mean_err"] = flux_mean_err
#
#     num_oh_median = []
#     num_oh_median_err = []
#     qs_median = []
#     qs_median_err = []
#     qs_median_max = []
#     qs_median_max_err = []
#     for _, row in df.iterrows():
#         num_oh = flux_OH_to_num_OH(
#             flux_OH=OHFlux(value=row.flux_median, sigma=row.flux_median_err),
#             helio_r_au=helio_r_au,
#             helio_v_kms=helio_v_kms,
#             delta_au=delta,
#         )
#         num_oh_median.append(num_oh.value)
#         num_oh_median_err.append(num_oh.sigma)
#
#         q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
#         qs_median.append(q.value)
#         qs_median_err.append(q.sigma)
#
#         num_oh_max = flux_OH_to_num_OH(
#             flux_OH=OHFlux(value=row.flux_median_max, sigma=row.flux_median_max_err),
#             helio_r_au=helio_r_au,
#             helio_v_kms=helio_v_kms,
#             delta_au=delta,
#         )
#         q_max = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh_max)
#         qs_median_max.append(q_max.value)
#         qs_median_max_err.append(q_max.sigma)
#
#     df["num_oh_median"] = num_oh_median
#     df["num_oh_median_err"] = num_oh_median_err
#     df["qs_median"] = qs_median
#     df["qs_median_err"] = qs_median_err
#     df["qs_median_max"] = qs_median_max
#     df["qs_median_max_err"] = qs_median_max_err
#
#     return df
