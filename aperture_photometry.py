import numpy as np
from photutils.aperture import (
    CircularAperture,
    ApertureStats,
    CircularAnnulus,
)

from dataclasses import dataclass
from typing import Dict


from swift_types import (
    SwiftFilter,
    filter_to_string,
    SwiftUVOTImage,
    SwiftStackedUVOTImage,
)

__version__ = "0.0.1"


__all__ = [
    "get_filter_parameters",
    "magnitude_from_count_rate",
    "determine_background",
    "do_aperture_photometry",
]


@dataclass
class AperturePhotometryResult:
    net_count: float
    net_count_rate: float
    source_count: float
    source_count_rate: float
    background_count: float
    background_count_rate: float


# TODO: cite these from the swift documentation
# TODO: look up what 'cf' stands for
# TODO: Make SwiftFilterParameters a dataclass?  Use typing.Final to make these constants?
# TODO: these are all technically a function of time, so we should incorporate that
def get_filter_parameters(filter_type: SwiftFilter) -> Dict:
    filter_params = {
        SwiftFilter.uvv: {
            "fwhm": 769,
            "zero_point": 17.89,
            "zero_point_err": 0.013,
            "cf": 2.61e-16,
            "cf_err": 2.4e-18,
        },
        SwiftFilter.ubb: {
            "fwhm": 975,
            "zero_point": 19.11,
            "zero_point_err": 0.016,
            "cf": 1.32e-16,
            "cf_err": 9.2e-18,
        },
        SwiftFilter.uuu: {
            "fwhm": 785,
            "zero_point": 18.34,
            "zero_point_err": 0.020,
            "cf": 1.5e-16,
            "cf_err": 1.4e-17,
        },
        SwiftFilter.uw1: {
            "fwhm": 693,
            "zero_point": 17.49,
            "zero_point_err": 0.03,
            "cf": 4.3e-16,
            "cf_err": 2.1e-17,
            "rf": 0.1375,
        },
        SwiftFilter.um2: {
            "fwhm": 498,
            "zero_point": 16.82,
            "zero_point_err": 0.03,
            "cf": 7.5e-16,
            "cf_err": 1.1e-17,
        },
        SwiftFilter.uw2: {
            "fwhm": 657,
            "zero_point": 17.35,
            "zero_point_err": 0.04,
            "cf": 6.0e-16,
            "cf_err": 6.4e-17,
        },
    }
    return filter_params[filter_type]


# TODO: error propogation
def magnitude_from_count_rate(count_rate, filter_type) -> float:
    filter_params = get_filter_parameters(filter_type)
    mag = filter_params["zero_point"] - 2.5 * np.log10(count_rate)
    # mag_err_1 = 2.5*cr_err/(np.log(10)*cr)
    # mag_err_2 = filt_para(filt)['zero_point_err']
    # mag_err = np.sqrt(mag_err_1**2 + mag_err_2**2)
    # return mag, mag_err
    return mag


def determine_background(
    image_data: SwiftUVOTImage,
    bg_aperture_x: float,
    bg_aperture_y: float,
    bg_aperture_radius: float,
) -> float:
    # inner_radius = 67
    # outer_radius = 123

    # image_center_row = np.ceil(image_data.shape[0] / 2)
    # image_center_col = np.ceil(image_data.shape[1] / 2)

    # comet_aperture = CircularAnnulus(
    #     (image_center_col, image_center_row),
    #     r_in=inner_radius,
    #     r_out=outer_radius,
    # )

    comet_aperture = CircularAperture(
        (bg_aperture_x, bg_aperture_y),
        r=bg_aperture_radius,
    )

    aperture_stats = ApertureStats(image_data, comet_aperture)
    return aperture_stats.median  # type: ignore

    # if we have passed in the median image from a filter, we can try this
    # return np.mean(image_data, axis=(0, 1))


def do_aperture_photometry(
    stacked_sum: SwiftStackedUVOTImage,
    stacked_median: SwiftStackedUVOTImage,
    comet_aperture_radius: float,
    bg_aperture_radius: float,
    bg_aperture_x: float,
    bg_aperture_y: float,
) -> AperturePhotometryResult:
    image_sum = stacked_sum.stacked_image
    image_median = stacked_median.stacked_image

    # print("\n")
    print(f"\nAperture photometry for {filter_to_string(stacked_sum.filter_type)}")
    print("---------------------------")
    # assume comet is centered in the stacked image, and that the stacked image has on odd number of pixels (the stacker should ensure this during stacking)
    image_center_row = np.ceil(image_sum.shape[0] / 2)
    image_center_col = np.ceil(image_sum.shape[1] / 2)
    print(f"Aperture center: {image_center_col}, {image_center_row}")

    # aperture radius: hard-coded
    # comet_aperture_radius = 32

    # reminder: x coords are columns, y rows
    comet_aperture = CircularAperture(
        (image_center_col, image_center_row), r=comet_aperture_radius
    )

    # use the aperture on the image
    aperture_stats = ApertureStats(image_sum, comet_aperture)
    print(f"Centroid of aperture: {aperture_stats.centroid}")

    # try using median-stacked image for getting the background
    background_count_rate_per_pixel = determine_background(
        image_median,
        bg_aperture_x=bg_aperture_x,
        bg_aperture_y=bg_aperture_y,
        bg_aperture_radius=bg_aperture_radius,
    )
    print(f"Background count rate per pixel: {background_count_rate_per_pixel}")
    print(
        f"Background counts per pixel: {background_count_rate_per_pixel * stacked_median.exposure_time}"
    )

    # the median images are in count rates, so multiply by exposure time for counts
    total_background_counts = (
        aperture_stats.sum_aper_area.value
        * background_count_rate_per_pixel
        * stacked_median.exposure_time
    )
    print(f"Background counts in aperture: {total_background_counts}")

    net_counts = aperture_stats.sum - total_background_counts
    print(f"Total counts in aperture, background corrected: {net_counts}")
    print(
        f"Count rate in aperture: {net_counts/stacked_sum.exposure_time} counts per second"
    )

    comet_magnitude = magnitude_from_count_rate(net_counts, stacked_sum.filter_type)
    print(f"Magnitude: {comet_magnitude}")

    return AperturePhotometryResult(
        net_count=net_counts,
        net_count_rate=net_counts / stacked_sum.exposure_time,
        source_count=aperture_stats.sum,  # type: ignore
        source_count_rate=aperture_stats.sum / stacked_sum.exposure_time,  # type: ignore
        background_count=total_background_counts,
        background_count_rate=total_background_counts / stacked_sum.exposure_time,
    )
