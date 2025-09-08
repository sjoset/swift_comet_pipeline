import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import numpy as np

from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.types.swift_filter import SwiftFilter


_seconds_in_a_year = 365.2425 * 86400.0

# Translate between filter type to the string in the CALDB FITS header key 'FILTER'
_swift_filter_to_caldb_dict = {
    SwiftFilter.uvv: "V",
    SwiftFilter.ubb: "B",
    SwiftFilter.uuu: "U",
    SwiftFilter.uw1: "UVW1",
    SwiftFilter.um2: "UVM2",
    SwiftFilter.uw2: "UVW2",
    SwiftFilter.white: "WHITE",
    SwiftFilter.magnifier: "MAGNIFIER",
}


# All times in the CALDB file are measured in seconds from this date
# TODO: rename this to MET, Mission Elapsed Time
def _get_uvot_sensitivity_start_date() -> Time:
    return Time("2005-01-01T00:00")


def _get_filter_sensitivity_data(filter_type: SwiftFilter) -> fits.FITS_rec | None:
    """
    Search through the extension headers until the 'FILTER' keyword matches the filter we want,
    and return the associated data table
    """

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    caldb_sensitivity_path = spc.uvot_sensitivity_path

    fits_header_filter_string = _swift_filter_to_caldb_dict.get(filter_type, None)
    if fits_header_filter_string is None:
        return None

    with fits.open(caldb_sensitivity_path) as hdulist:
        for hdu in hdulist:
            hdr = hdu.header  # type: ignore
            if hdr.get("FILTER") == fits_header_filter_string:
                return hdu.data.copy()  # type: ignore

    return None


def _seconds_since_uvot_sensitivity_start_date(t: Time) -> float:

    time_delta = t - _get_uvot_sensitivity_start_date()

    # time_delta_s: float = time_delta.to_value(u.s)  # type: ignore
    return time_delta.to_value(u.s)  # type: ignore


def uvot_sensitivity_correction_factor(
    filter_type: SwiftFilter, t_obs: Time
) -> float | None:

    sensitivity_table = _get_filter_sensitivity_data(filter_type=filter_type)
    if sensitivity_table is None:
        return None

    t_obs_delta = _seconds_since_uvot_sensitivity_start_date(t=t_obs)
    if t_obs_delta < 0:
        return None

    # Table columns are double(TIME), float(OFFSET), float(SLOPE)
    times_since_start = sensitivity_table["TIME"].astype(float)  # type: ignore
    offsets = sensitivity_table["OFFSET"].astype(float)  # type: ignore
    slopes = sensitivity_table["SLOPE"].astype(float)  # type: ignore

    # latest row with TIME <= t_mid
    idx = np.where(times_since_start <= t_obs_delta)[0]

    # observation time predates CALDB range: no correction factors available for those times
    if idx.size == 0:
        return 1.0

    i = idx.max()
    dt_yrs = (t_obs_delta - times_since_start[i]) / _seconds_in_a_year
    correction_factor = (1.0 + offsets[i]) * (1.0 + slopes[i]) ** dt_yrs

    return float(correction_factor)
