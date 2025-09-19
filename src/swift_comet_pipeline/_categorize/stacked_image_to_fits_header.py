from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.scp_types.primitive import *

from swift_comet_pipeline.scp_types.compound.epoch_summary import EpochSummary
from swift_comet_pipeline.images.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.swift.filters.uvot_filter_to_string import (
    filter_to_file_string,
)


# TODO: relocate this function
def epoch_stacked_image_to_fits(
    epoch_summary: EpochSummary, img: SwiftUvotImage, filter_type: UvotFilter
) -> fits.ImageHDU:
    """
    Takes the image and fills out a FITS header
    Assumes the image is centered on the comet
    """

    hdu = fits.ImageHDU(data=img)

    # TODO: include data mode or event mode here, time of processing, pipeline version?

    hdr = hdu.header
    hdr["distunit"] = "AU"
    hdr["v_unit"] = "km/s"
    hdr["delta"] = epoch_summary.delta_au
    hdr["rh"] = epoch_summary.rh_au
    if filter_type == UvotFilter.uw1:
        exp_time = epoch_summary.uw1_exposure_time_s
    elif filter_type == UvotFilter.uvv:
        exp_time = epoch_summary.uvv_exposure_time_s
    else:
        exp_time = 0.0
    hdr["exposure_time_s"] = exp_time
    hdr["filter"] = filter_to_file_string(filter_type=filter_type)
    hdr["epoch_id"] = epoch_summary.epoch_id
    hdr["sky_motion_arcsec_min"] = epoch_summary.sky_motion_arcsec_min
    hdr["time_from_perihelion_days"] = epoch_summary.time_from_perihelion.to_value(
        u.day  # type: ignore
    )
    hdr["observation_time"] = str(Time(epoch_summary.observation_time))
    hdr["epoch_length_seconds"] = epoch_summary.epoch_length.total_seconds()
    hdr["helio_v_kms"] = epoch_summary.helio_v_kms
    hdr["phase"] = epoch_summary.phase_angle_deg

    pix_center = get_uvot_image_center(img=img)
    hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y

    return hdu
