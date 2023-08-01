import numpy as np
from photutils.aperture import (
    CircularAperture,
    ApertureStats,
    # CircularAnnulus,
)

from dataclasses import dataclass
from typing import Dict


from swift_types import (
    SwiftFilter,
    filter_to_string,
    SwiftUVOTImage,
    # SwiftStackedUVOTImage,
)

__version__ = "0.0.1"


__all__ = [
    "get_filter_parameters",
    "magnitude_from_count_rate",
    "do_aperture_photometry",
]


# @dataclass
# class AperturePhotometryResult:
#     net_counts: float
#     net_count_rate: float
#     comet_counts: float
#     comet_count_rate: float
#     background_counts_in_comet_aperture: float
#     background_count_rate_in_comet_aperture: float
#     background_counts: float
#     background_count_rate: float
#     comet_aperture: CircularAperture
#     background_aperture: CircularAperture
#     comet_magnitude: float


# # TODO: move to another file
# # TODO: cite these from the swift documentation
# # TODO: look up what 'cf' stands for
# # TODO: Make SwiftFilterParameters a dataclass?  Use typing.Final to make these constants?
# # TODO: these are all technically a function of time, so we should incorporate that
# def get_filter_parameters(filter_type: SwiftFilter) -> Dict:
#     filter_params = {
#         SwiftFilter.uvv: {
#             "fwhm": 769,
#             "zero_point": 17.89,
#             "zero_point_err": 0.013,
#             "cf": 2.61e-16,
#             "cf_err": 2.4e-18,
#         },
#         SwiftFilter.ubb: {
#             "fwhm": 975,
#             "zero_point": 19.11,
#             "zero_point_err": 0.016,
#             "cf": 1.32e-16,
#             "cf_err": 9.2e-18,
#         },
#         SwiftFilter.uuu: {
#             "fwhm": 785,
#             "zero_point": 18.34,
#             "zero_point_err": 0.020,
#             "cf": 1.5e-16,
#             "cf_err": 1.4e-17,
#         },
#         SwiftFilter.uw1: {
#             "fwhm": 693,
#             "zero_point": 17.49,
#             "zero_point_err": 0.03,
#             "cf": 4.3e-16,
#             "cf_err": 2.1e-17,
#             "rf": 0.1375,
#         },
#         SwiftFilter.um2: {
#             "fwhm": 498,
#             "zero_point": 16.82,
#             "zero_point_err": 0.03,
#             "cf": 7.5e-16,
#             "cf_err": 1.1e-17,
#         },
#         SwiftFilter.uw2: {
#             "fwhm": 657,
#             "zero_point": 17.35,
#             "zero_point_err": 0.04,
#             "cf": 6.0e-16,
#             "cf_err": 6.4e-17,
#         },
#     }
#     return filter_params[filter_type]
#
#
# # TODO: move to another file
# # TODO: error propogation
# def magnitude_from_count_rate(count_rate, filter_type) -> float:
#     filter_params = get_filter_parameters(filter_type)
#     mag = filter_params["zero_point"] - 2.5 * np.log10(count_rate)
#     # mag_err_1 = 2.5*cr_err/(np.log(10)*cr)
#     # mag_err_2 = filt_para(filt)['zero_point_err']
#     # mag_err = np.sqrt(mag_err_1**2 + mag_err_2**2)
#     # return mag, mag_err
#     return mag


# TODO: this should also take a time of observation, because the magnitude calculation uses
# filter data that is time dependent
def do_aperture_photometry(
    stacked_sum: SwiftUVOTImage,
    stacked_median: SwiftUVOTImage,
    comet_aperture_radius: float,
    bg_aperture_radius: float,
    bg_aperture_x: float,
    bg_aperture_y: float,
) -> AperturePhotometryResult:
    image_sum = stacked_sum.stacked_image
    image_median = stacked_median.stacked_image
    exposure_time = stacked_sum.exposure_time

    # print("\n")
    print(f"\nAperture photometry for {filter_to_string(stacked_sum.filter_type)}")
    print("---------------------------")

    image_center_row = int(np.floor(image_sum.shape[0] / 2))
    image_center_col = int(np.floor(image_sum.shape[1] / 2))
    print(
        f"Using test aperture at image center: {image_center_col}, {image_center_row}"
    )

    # reminder: x coords are columns, y rows
    initial_comet_aperture = CircularAperture(
        (image_center_col, image_center_row), r=comet_aperture_radius
    )

    # use an aperture on the image center of the specified radius and find the centroid of the comet signal
    initial_aperture_stats = ApertureStats(image_sum, initial_comet_aperture)
    print(
        f"Moving analysis aperture to the centroid of the test aperture: {initial_aperture_stats.centroid}"
    )

    # Move the aperture to the centroid of the test aperture and do our analysis there
    comet_center_x, comet_center_y = (
        initial_aperture_stats.centroid[0],
        initial_aperture_stats.centroid[1],
    )
    comet_aperture = CircularAperture(
        (comet_center_x, comet_center_y), r=comet_aperture_radius
    )
    comet_aperture_stats = ApertureStats(image_sum, comet_aperture)
    print(f"Centroid of analysis aperture: {comet_aperture_stats.centroid}")

    comet_count_rate = float(comet_aperture_stats.sum)
    # the sum images are in count rates, so multiply by exposure time for counts
    comet_counts = comet_count_rate * exposure_time

    # try using median-stacked image for getting the background
    background_aperture = CircularAperture(
        (bg_aperture_x, bg_aperture_y),
        r=bg_aperture_radius,
    )

    background_aperture_stats = ApertureStats(image_median, background_aperture)
    background_count_rate_per_pixel = background_aperture_stats.median  # type: ignore

    # the median images are in count rates, so multiply by exposure time for counts
    background_counts = (
        background_aperture_stats.sum_aper_area.value
        * background_count_rate_per_pixel
        * exposure_time
    )
    background_count_rate = background_counts / exposure_time
    print(f"Background counts in aperture: {background_counts}")
    print(f"Background count rate per pixel: {background_count_rate_per_pixel}")
    print(f"Background aperture area: {background_aperture_stats.sum_aper_area} ")

    background_count_rate_in_comet_aperture = (
        background_count_rate_per_pixel * comet_aperture_stats.sum_aper_area.value
    )
    background_counts_in_comet_aperture = (
        background_count_rate_in_comet_aperture * exposure_time
    )
    net_counts = comet_counts - background_counts_in_comet_aperture
    net_count_rate = net_counts / exposure_time
    print(f"Net counts in aperture: {net_counts}")
    print(f"Net count rate in aperture: {net_count_rate} counts per second")

    comet_magnitude = magnitude_from_count_rate(net_count_rate, stacked_sum.filter_type)
    print(f"Magnitude: {comet_magnitude}")

    return AperturePhotometryResult(
        net_counts=net_counts,
        net_count_rate=net_count_rate,
        comet_counts=comet_counts,
        comet_count_rate=comet_count_rate,
        background_counts_in_comet_aperture=background_counts_in_comet_aperture,
        background_count_rate_in_comet_aperture=background_count_rate_in_comet_aperture,
        background_counts=background_counts,
        background_count_rate=background_count_rate,
        comet_aperture=comet_aperture,
        background_aperture=background_aperture,
        comet_magnitude=comet_magnitude,
    )
