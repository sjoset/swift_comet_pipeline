from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.swift.filters.uvot_filter_to_string import (
    filter_to_file_string,
)


def stacked_image_to_fits(
    epoch_index_entry: EpochIndexEntry, img: SwiftUvotImage, filter_type: UvotFilter
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
    hdr["delta"] = epoch_index_entry.delta_au
    hdr["rh"] = epoch_index_entry.rh_au
    hdr["exposure_time_s"] = epoch_index_entry.exposure_times[filter_type]
    hdr["filter"] = filter_to_file_string(filter_type=filter_type)
    hdr["epoch_id"] = epoch_index_entry.epoch_id
    hdr["sky_motion_arcsec_min"] = epoch_index_entry.sky_motion_arcsec_min
    hdr["time_from_perihelion_days"] = epoch_index_entry.time_from_perihelion.to_value(
        u.day  # type: ignore
    )
    hdr["observation_time"] = str(Time(epoch_index_entry.observation_time))
    hdr["epoch_length_seconds"] = epoch_index_entry.epoch_length.total_seconds()
    hdr["helio_v_kms"] = epoch_index_entry.helio_v_kms
    hdr["phase"] = epoch_index_entry.phase_angle_deg

    pix_center = get_uvot_image_center(img=img)
    hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y

    return hdu
