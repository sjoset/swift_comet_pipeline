import pathlib
import numpy as np
import pandas as pd

from typing import TypeAlias
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.observationlog.observation_log import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage, get_uvot_image_center


Epoch: TypeAlias = pd.DataFrame
EpochID: TypeAlias = str


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


def is_epoch_stackable(epoch: Epoch) -> bool:
    """
    Checks that all uw1 and uvv images in this epoch are taken with the same DATAMODE keyword
    """
    return epoch.DATAMODE.nunique() == 1


def epoch_stacked_image_to_fits(epoch: Epoch, img: SwiftUVOTImage) -> fits.ImageHDU:
    # TODO: relocate this function

    hdu = fits.ImageHDU(data=img)

    # TODO: include data mode or event mode here, time of processing, pipeline version?

    hdr = hdu.header
    hdr["distunit"] = "AU"
    hdr["v_unit"] = "km/s"
    hdr["delta"] = np.mean(epoch.OBS_DIS)
    hdr["rh"] = np.mean(epoch.HELIO)
    hdr["ra_obj"] = np.mean(epoch.RA_OBJ)
    hdr["dec_obj"] = np.mean(epoch.DEC_OBJ)

    # TODO: read epoch for center info in case user changed it
    pix_center = get_uvot_image_center(img=img)
    hdr["pos_x"], hdr["pos_y"] = pix_center.x, pix_center.y
    hdr["phase"] = np.mean(epoch.PHASE)

    dt = Time(np.max(epoch.MID_TIME)) - Time(np.min(epoch.MID_TIME))
    first_obs_row = epoch.loc[epoch.MID_TIME.idxmin()]
    last_obs_row = epoch.loc[epoch.MID_TIME.idxmax()]

    first_obs_time = Time(first_obs_row.MID_TIME)
    first_obs_time.format = "fits"
    hdr["firstobs"] = first_obs_time.value
    last_obs_time = Time(last_obs_row.MID_TIME)
    last_obs_time.format = "fits"
    hdr["lastobs"] = last_obs_time.value
    mid_obs = Time(np.mean(epoch.MID_TIME))
    mid_obs.format = "fits"
    hdr["mid_obs"] = mid_obs.value

    rh_start = first_obs_row.HELIO * u.AU  # type: ignore
    rh_end = last_obs_row.HELIO * u.AU  # type: ignore
    dr_dt = (rh_end - rh_start) / dt

    ddelta_dt = (last_obs_row.OBS_DIS * u.AU - first_obs_row.OBS_DIS * u.AU) / dt  # type: ignore

    hdr["drh_dt"] = dr_dt.to_value(u.km / u.s)  # type: ignore
    hdr["ddeltadt"] = ddelta_dt.to_value(u.km / u.s)  # type: ignore

    return hdu
