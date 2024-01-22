from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture
from scipy.integrate import simpson
from swift_comet_pipeline.epochs import Epoch
from swift_comet_pipeline.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.flux_OH import OH_flux_from_count_rate, OHFlux
from swift_comet_pipeline.num_OH_to_Q import num_OH_to_Q_vectorial
from swift_comet_pipeline.uvot_image import (
    PixelCoord,
    SwiftUVOTImage,
    get_uvot_image_center,
)
from swift_comet_pipeline.count_rate import CountRate, CountRatePerPixel


__all__ = [
    "CometRadialProfile",
    "CometProfile",
    "extract_comet_radial_profile",
    "count_rate_from_comet_radial_profile",
    "count_rate_from_comet_profile",
    "surface_brightness_profiles",
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

    return CountRate(value=float(count_rate), sigma=propogated_sigma)


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

    return CountRate(value=float(count_rate), sigma=propogated_sigma)


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
    # TODO: use calc_total_error here
    comet_count_rate_sigma = comet_aperture_stats.std

    propogated_sigma = np.sqrt(
        comet_count_rate_sigma**2
        + (comet_aperture_stats.sum_aper_area.value * bg.sigma**2)
    )

    return CountRate(value=comet_count_rate, sigma=propogated_sigma)


@dataclass
class SurfaceBrightnessProfiles:
    r_inner_pix: np.ndarray
    r_outer_pix: np.ndarray


def surface_brightness_profiles(
    uw1: SwiftUVOTImage, uvv: SwiftUVOTImage, r_max: int
) -> pd.DataFrame:
    # fix the annular aperture thickness at one pixel
    radial_slice_thickness: int = 1

    comet_peak = get_uvot_image_center(uw1)
    # first slice is a circular aperture centered on the comet, after which we move to annular apertures
    center_aperture = CircularAperture(
        positions=(comet_peak.x, comet_peak.y), r=radial_slice_thickness
    )
    center_uw1_stats = ApertureStats(data=uw1, aperture=center_aperture)
    center_uvv_stats = ApertureStats(data=uvv, aperture=center_aperture)

    # construct annular apertures
    r_inner_max = r_max - radial_slice_thickness
    num_apertures = r_max - radial_slice_thickness
    inner_aperture_rs = np.linspace(
        start=radial_slice_thickness, stop=r_inner_max, endpoint=True, num=num_apertures
    )
    annular_apertures = [
        CircularAnnulus((comet_peak.x, comet_peak.y), r_in=r_inner, r_out=r_inner + 1)
        for r_inner in inner_aperture_rs
    ]

    uw1_stats = [ApertureStats(data=uw1, aperture=ap) for ap in annular_apertures]
    uvv_stats = [ApertureStats(data=uvv, aperture=ap) for ap in annular_apertures]

    # add results to the dataframe, by pasting together the center results with the results from
    # the aperture list
    df = pd.DataFrame()
    df["r_inner_pix"] = np.append(np.array([0]), inner_aperture_rs)
    df["r_outer_pix"] = df["r_inner_pix"] + 1
    df["surface_brightness_uw1_median"] = np.append(
        center_uw1_stats.median, [x.median for x in uw1_stats]
    )
    df["surface_brightness_uw1_median_err"] = np.append(
        center_uw1_stats.std, [1.2533 * x.std for x in uw1_stats]
    )
    df["surface_brightness_uw1_mean"] = np.append(
        center_uw1_stats.mean, [x.mean for x in uw1_stats]
    )
    df["surface_brightness_uw1_mean_err"] = np.append(
        center_uw1_stats.std, [x.std for x in uw1_stats]
    )
    df["surface_brightness_uvv_median"] = np.append(
        center_uvv_stats.median, [x.median for x in uvv_stats]
    )
    df["surface_brightness_uvv_median_err"] = np.append(
        center_uvv_stats.std, [1.2533 * x.std for x in uvv_stats]
    )
    df["surface_brightness_uvv_mean"] = np.append(
        center_uvv_stats.mean, [x.mean for x in uvv_stats]
    )
    df["surface_brightness_uvv_mean_err"] = np.append(
        center_uvv_stats.std, [x.std for x in uvv_stats]
    )

    df["cumulative_counts_uw1_mean"] = df.surface_brightness_uw1_mean.cumsum()
    # df["cumulative_counts_uw1_mean_err1"] = df.surface_brightness_uw1_mean_err.cumsum()
    df["cumulative_counts_uw1_mean_err"] = np.sqrt(
        np.cumsum(np.square(df.cumulative_counts_uw1_mean))
    )
    df["cumulative_counts_uw1_median"] = df.surface_brightness_uw1_median.cumsum()
    # TODO: finish converting error to quadrature
    # df[
    #     "cumulative_counts_uw1_median_err"
    # ] = df.surface_brightness_uw1_median_err.cumsum()
    df["cumulative_counts_uw1_median_err"] = np.sqrt(
        np.cumsum(np.square(df.cumulative_counts_uw1_median))
    )
    df["cumulative_counts_uvv_mean"] = df.surface_brightness_uvv_mean.cumsum()
    # df["cumulative_counts_uvv_mean_err"] = df.surface_brightness_uvv_mean_err.cumsum()
    df["cumulative_counts_uvv_mean_err"] = np.sqrt(
        np.cumsum(np.square(df.cumulative_counts_uvv_mean))
    )
    df["cumulative_counts_uvv_median"] = df.surface_brightness_uvv_median.cumsum()
    # df[
    #     "cumulative_counts_uvv_median_err"
    # ] = df.surface_brightness_uvv_median_err.cumsum()
    df["cumulative_counts_uvv_median_err"] = np.sqrt(
        np.cumsum(np.square(df.cumulative_counts_uvv_median))
    )

    return df


def qh2o_from_surface_brightness_profiles(
    df: pd.DataFrame, epoch: Epoch, beta: float
) -> pd.DataFrame:
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    df["oh_brightness_median"] = (
        df.surface_brightness_uw1_median - beta * df.surface_brightness_uvv_median
    )
    df["oh_brightness_mean"] = (
        df.surface_brightness_uw1_mean - beta * df.surface_brightness_uvv_mean
    )
    df["oh_brightness_running_total_median"] = df.oh_brightness_median.cumsum()
    df["oh_brightness_running_total_mean"] = df.oh_brightness_mean.cumsum()

    flux_median = []
    flux_median_err = []
    flux_median_max = []
    flux_median_max_err = []
    for _, row in df.iterrows():
        uw1cr = CountRate(
            value=row.cumulative_counts_uw1_median,
            sigma=row.cumulative_counts_uw1_median_err,
        )
        uvvcr = CountRate(
            value=row.cumulative_counts_uvv_median,
            sigma=row.cumulative_counts_uvv_median_err,
        )
        flux_OH = OH_flux_from_count_rate(uw1=uw1cr, uvv=uvvcr, beta=beta)
        flux_median.append(flux_OH.value)
        flux_median_err.append(flux_OH.sigma)

        flux_max = OH_flux_from_count_rate(
            uw1=uw1cr, uvv=CountRate(value=0.0, sigma=0.0), beta=beta
        )
        flux_median_max.append(flux_max.value)
        flux_median_max_err.append(flux_max.sigma)

    df["flux_median"] = flux_median
    df["flux_median_err"] = flux_median_err
    df["flux_median_max"] = flux_median_max
    df["flux_median_max_err"] = flux_median_max_err

    flux_mean = []
    flux_mean_err = []
    for _, row in df.iterrows():
        uw1cr = CountRate(
            value=row.cumulative_counts_uw1_mean,
            sigma=row.cumulative_counts_uw1_mean_err,
        )
        uvvcr = CountRate(
            value=row.cumulative_counts_uvv_mean,
            sigma=row.cumulative_counts_uvv_mean_err,
        )
        flux_OH = OH_flux_from_count_rate(uw1=uw1cr, uvv=uvvcr, beta=beta)
        flux_mean.append(flux_OH.value)
        flux_mean_err.append(flux_OH.sigma)

    df["flux_mean"] = flux_mean
    df["flux_mean_err"] = flux_mean_err

    num_oh_median = []
    num_oh_median_err = []
    qs_median = []
    qs_median_err = []
    qs_median_max = []
    qs_median_max_err = []
    for _, row in df.iterrows():
        num_oh = flux_OH_to_num_OH(
            flux_OH=OHFlux(value=row.flux_median, sigma=row.flux_median_err),
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta,
        )
        num_oh_median.append(num_oh.value)
        num_oh_median_err.append(num_oh.sigma)

        q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
        qs_median.append(q.value)
        qs_median_err.append(q.sigma)

        num_oh_max = flux_OH_to_num_OH(
            flux_OH=OHFlux(value=row.flux_median_max, sigma=row.flux_median_max_err),
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta,
        )
        q_max = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh_max)
        qs_median_max.append(q_max.value)
        qs_median_max_err.append(q_max.sigma)

    df["num_oh_median"] = num_oh_median
    df["num_oh_median_err"] = num_oh_median_err
    df["qs_median"] = qs_median
    df["qs_median_err"] = qs_median_err
    df["qs_median_max"] = qs_median_max
    df["qs_median_max_err"] = qs_median_max_err

    return df


# def _surface_brightness_profile(swift_project_config: SwiftProjectConfig) -> None:
#     pipeline_files = PipelineFiles(
#         base_product_save_path=swift_project_config.product_save_path
#     )
#
#     # select the epoch we want to process
#     epoch_id = stacked_epoch_menu(
#         pipeline_files=pipeline_files, require_background_analysis_to_be=True
#     )
#     if epoch_id is None:
#         return
#
#     # load the epoch database
#     epoch = pipeline_files.read_pipeline_product(
#         PipelineProductType.stacked_epoch, epoch_id=epoch_id
#     )
#     epoch_path = pipeline_files.get_product_path(
#         PipelineProductType.stacked_epoch, epoch_id=epoch_id
#     )
#     if epoch is None or epoch_path is None:
#         print("Error loading epoch!")
#         wait_for_key()
#         return
#
#     print(
#         f"Starting analysis of {epoch_path.stem}: observation at {np.mean(epoch.HELIO)} AU"
#     )
#
#     stacking_method = StackingMethod.summation
#
#     # load background-subtracted images
#     uw1 = pipeline_files.read_pipeline_product(
#         PipelineProductType.background_subtracted_image,
#         epoch_id=epoch_id,
#         filter_type=SwiftFilter.uw1,
#         stacking_method=stacking_method,
#     )
#     uvv = pipeline_files.read_pipeline_product(
#         PipelineProductType.background_subtracted_image,
#         epoch_id=epoch_id,
#         filter_type=SwiftFilter.uvv,
#         stacking_method=stacking_method,
#     )
#     if uw1 is None or uvv is None:
#         print("Error loading background-subtracted images!")
#         wait_for_key()
#         return
#
#     uw1_exposure_time = np.sum(epoch[epoch.FILTER == SwiftFilter.uw1].EXPOSURE)
#     uvv_exposure_time = np.sum(epoch[epoch.FILTER == SwiftFilter.uvv].EXPOSURE)
#
#     comet_peak = get_uvot_image_center(uw1)
#
#     center_aperture = CircularAperture(positions=(comet_peak.x, comet_peak.y), r=1)
#     center_uw1_stats = ApertureStats(data=uw1, aperture=center_aperture)
#     center_uvv_stats = ApertureStats(data=uvv, aperture=center_aperture)
#
#     r_inner_max = 199
#     inner_aperture_rs = np.linspace(1, r_inner_max, endpoint=True, num=r_inner_max)
#     # r_kms = inner_aperture_rs * np.mean(epoch.KM_PER_PIX)
#
#     annular_apertures = [
#         CircularAnnulus((comet_peak.x, comet_peak.y), r_in=r_inner, r_out=r_inner + 1)
#         for r_inner in inner_aperture_rs
#     ]
#
#     uw1_stats = [ApertureStats(data=uw1, aperture=ap) for ap in annular_apertures]
#     uvv_stats = [ApertureStats(data=uvv, aperture=ap) for ap in annular_apertures]
#
#     beta = beta_parameter(DustReddeningPercent(50.0))
#
#     # median_d = center_uw1_stats.median - beta * center_uvv_stats.median
#     # mean_d = center_uw1_stats.mean - beta * center_uvv_stats.mean
#
#     # rprint(
#     #     f"0 to 1\t\t\t[blue]{center_uw1_stats.median}\t{center_uw1_stats.mean}[/blue]\t[purple]{center_uvv_stats.median}\t{center_uvv_stats.mean}[/purple]\t[red]{median_d}\t{mean_d}[/red]",
#     # )
#     # for r, r_km, uw1_stat, uvv_stat in zip(
#     #     inner_aperture_rs, r_kms, uw1_stats, uvv_stats
#     # ):
#     #     mean_diff = uw1_stat.mean - beta * uvv_stat.mean
#     #     median_diff = uw1_stat.median - beta * uvv_stat.median
#     #     rprint(
#     #         f"{r=:03.0f} ({r_km:3.2e}) to {r+1:03.0f}\t[blue]{uw1_stat.median}\t{uw1_stat.mean}[/blue]\t[purple]{uvv_stat.median}\t{uvv_stat.mean}[/purple]\t[red]{median_diff}\t{mean_diff}[/red]",
#     #     )
#
#     df = pd.DataFrame()
#     df["r_inner_pix"] = np.append(np.array([0]), inner_aperture_rs)
#     df["r_inner_km"] = df.r_inner_pix * np.mean(epoch.KM_PER_PIX)
#     df["r_outer_pix"] = df["r_inner_pix"] + 1
#     df["r_outer_km"] = df.r_outer_pix * np.mean(epoch.KM_PER_PIX)
#     df["r_mid_km"] = (df.r_outer_km + df.r_inner_km) / 2
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
#     df["cumulative_counts_uw1_mean"] = df.surface_brightness_uw1_mean.cumsum()
#     df["cumulative_counts_uw1_mean_err"] = df.surface_brightness_uw1_mean_err.cumsum()
#     df["cumulative_counts_uw1_median"] = df.surface_brightness_uw1_median.cumsum()
#     df[
#         "cumulative_counts_uw1_median_err"
#     ] = df.surface_brightness_uw1_median_err.cumsum()
#     df["cumulative_counts_uvv_mean"] = df.surface_brightness_uvv_mean.cumsum() * beta
#     df["cumulative_counts_uvv_mean_err"] = (
#         df.surface_brightness_uvv_mean_err.cumsum() * beta
#     )
#     df["cumulative_counts_uvv_median"] = (
#         df.surface_brightness_uvv_median.cumsum() * beta
#     )
#     df[
#         "cumulative_counts_uvv_median_err"
#     ] = df.surface_brightness_uvv_median_err.cumsum()
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
#     df["flux_median"] = flux_median
#     df["flux_median_err"] = flux_median_err
#     # df["oh_flux_median"] = OH_flux_from_count_rate(
#     #     uw1=df.cumulative_counts_uw1_median,
#     #     uvv=df.cumulative_counts_uvv_median,
#     #     beta=beta,
#     # )
#
#     print(df)
#
#     # for c in ["cumulative_counts_uw1_mean", "cumulative_counts_uvv_mean"]:
#     for c in ["cumulative_counts_uw1_mean"]:
#         df[c].plot()
#     plt.show()
#
#     for c in ["cumulative_counts_uw1_median", "cumulative_counts_uvv_median"]:
#         df[c].plot()
#     plt.show()
#
#     wait_for_key()
