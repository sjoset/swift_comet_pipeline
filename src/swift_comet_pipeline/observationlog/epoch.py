import pathlib

from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.observationlog.epoch_typing import Epoch
from swift_comet_pipeline.observationlog.observation_log import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.swift.swift_filter_to_string import filter_to_file_string
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def read_epoch(epoch_path: pathlib.Path) -> Epoch:
    """
    Allow read_observation_log to do post-load processing on SwiftObservationLog columns
    """
    epoch = read_observation_log(epoch_path)

    # do any column processing of our own here

    return epoch


def write_epoch(epoch: Epoch, epoch_path: pathlib.Path) -> None:
    # schema = epoch_schema()
    # if additional_schema is not None:
    #     schema = pa.unify_schemas([schema, additional_schema])

    # do any column processing of our own here

    write_observation_log(epoch, epoch_path)


def epoch_stacked_image_to_fits(
    epoch_summary: EpochSummary, img: SwiftUVOTImage, filter_type: SwiftFilter
) -> fits.ImageHDU:
    # TODO: relocate this function
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
    if filter_type == SwiftFilter.uw1:
        exp_time = epoch_summary.uw1_exposure_time_s
    elif filter_type == SwiftFilter.uvv:
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
    # hdr["pixel_resolution_arcsec"] = epoch_summary.pixel_resolution.value
    hdr["observation_time"] = str(Time(epoch_summary.observation_time))
    hdr["epoch_length_seconds"] = epoch_summary.epoch_length.total_seconds()
    hdr["helio_v_kms"] = epoch_summary.helio_v_kms
    hdr["phase"] = epoch_summary.phase_angle_deg

    pix_center = get_uvot_image_center(img=img)
    hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y

    return hdu


# def epoch_stacked_image_to_fits(epoch: Epoch, img: SwiftUVOTImage) -> fits.ImageHDU:
#     # TODO: relocate this function and rewrite for EpochSummary
#
#     hdu = fits.ImageHDU(data=img)
#
#     # TODO: include data mode or event mode here, time of processing, pipeline version?
#
#     hdr = hdu.header
#     hdr["distunit"] = "AU"
#     hdr["v_unit"] = "km/s"
#     hdr["delta"] = np.mean(epoch.OBS_DIS)
#     hdr["rh"] = np.mean(epoch.HELIO)
#     hdr["ra_obj"] = np.mean(epoch.RA_OBJ)
#     hdr["dec_obj"] = np.mean(epoch.DEC_OBJ)
#
#     # TODO: read epoch for center info in case user changed it
#     pix_center = get_uvot_image_center(img=img)
#     hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y
#     hdr["phase"] = np.mean(epoch.PHASE)
#
#     dt = Time(np.max(epoch.MID_TIME)) - Time(np.min(epoch.MID_TIME))
#     first_obs_row = epoch.loc[epoch.MID_TIME.idxmin()]
#     last_obs_row = epoch.loc[epoch.MID_TIME.idxmax()]
#
#     first_obs_time = Time(first_obs_row.MID_TIME)
#     first_obs_time.format = "fits"
#     hdr["firstobs"] = first_obs_time.value
#     last_obs_time = Time(last_obs_row.MID_TIME)
#     last_obs_time.format = "fits"
#     hdr["lastobs"] = last_obs_time.value
#     mid_obs = Time(np.mean(epoch.MID_TIME))
#     mid_obs.format = "fits"
#     hdr["mid_obs"] = mid_obs.value
#
#     rh_start = first_obs_row.HELIO * u.AU  # type: ignore
#     rh_end = last_obs_row.HELIO * u.AU  # type: ignore
#     dr_dt = (rh_end - rh_start) / dt
#
#     ddelta_dt = (last_obs_row.OBS_DIS * u.AU - first_obs_row.OBS_DIS * u.AU) / dt  # type: ignore
#
#     hdr["drh_dt"] = dr_dt.to_value(u.km / u.s)  # type: ignore
#     hdr["ddeltadt"] = ddelta_dt.to_value(u.km / u.s)  # type: ignore
#
#     return hdu
