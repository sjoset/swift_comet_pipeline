import astropy.units as u

from swift_comet_pipeline.post_pipeline_analysis.column_density_above_background import (
    ColumnDensityAboveBackgroundAnalysis,
)


def column_density_has_enough_coverage(
    cd_bg: ColumnDensityAboveBackgroundAnalysis,
    min_distance_coverage: u.Quantity = 100_000 * u.km,  # type: ignore
) -> bool:

    if cd_bg.last_usable_r < min_distance_coverage:
        return False

    return True


def column_density_larger_than_psf_threshold(
    cd_bg: ColumnDensityAboveBackgroundAnalysis, num_psfs_required: float = 3.0
) -> bool:

    # TODO: psf radius of 5 is hardcoded: is there a better way to do this?  Make psf part of the swift module?
    swift_psf_radius = 5 / cd_bg.pixel_resolution
    if cd_bg.num_usable_pixels_in_profile < (num_psfs_required * swift_psf_radius):
        return False

    return True


def vectorial_fitting_reliable(
    cd_bg: ColumnDensityAboveBackgroundAnalysis,
    min_distance_coverage: u.Quantity = 100_000 * u.km,  # type: ignore
    num_psfs_required: float = 3.0,
) -> bool:
    """
    For vectorial fitting to be reliable, it needs to have information covering roughly min_distance_coverage (100,000 km),
    and the profile must extend beyond num_psfs_required (default 3), otherwise the column density is too smeared by the psf
    """

    if not column_density_has_enough_coverage(
        cd_bg=cd_bg, min_distance_coverage=min_distance_coverage
    ) or not column_density_larger_than_psf_threshold(
        cd_bg=cd_bg, num_psfs_required=num_psfs_required
    ):
        return False

    return True
