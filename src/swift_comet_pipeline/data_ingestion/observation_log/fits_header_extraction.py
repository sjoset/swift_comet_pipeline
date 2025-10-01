import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


from swift_comet_pipeline.scp_types.primitive import *

from swift_comet_pipeline.scp_types.compound.swift_level_2_fits import (
    SwiftLevel2FITSObservation,
)


def is_fits_image_hdu(hdu) -> bool:
    return isinstance(hdu, fits.ImageHDU)


def is_event_mode_bintable(hdu) -> bool:
    return isinstance(hdu, fits.BinTableHDU)


_level_2_fits_header_keys_to_extract = [
    "OBS_ID",
    "DATE-OBS",
    "DATE-END",
    "FILTER",
    "RA_OBJ",
    "DEC_OBJ",
    "EXPOSURE",
    "DATAMODE",
    "CREATOR",
]


def _level_2_data_mode_fits_header_to_columns(
    obs: SwiftLevel2FITSObservation,
) -> list[pd.Series] | None:
    """
    For the given observation, pull all keys in _level_2_fits_header_keys_to_extract from the image header
    and fill in columns of a Series with the same names as the keys
    Coerces to proper types
    """

    series_list = []

    with fits.open(obs.fits_path) as hdul:
        for extension_index, hdu in enumerate(hdul):
            # skip the first extension, which should be informational
            if extension_index == 0:
                continue

            # check if this extension is an image
            if not is_fits_image_hdu(hdu=hdu):
                print(
                    f"Skipping extension {extension_index} of {obs.fits_path}: not an image HDU"
                )
                continue

            header_series = {
                k: hdu.header.get(k, None) for k in _level_2_fits_header_keys_to_extract
            }
            header_series["WCS"] = WCS(hdu.header)
            header_series["EXTENSION"] = extension_index
            header_series["FITS_FILENAME"] = str(obs.fits_path.name)
            header_series["FULL_FITS_PATH"] = obs.fits_path
            series_list.append(pd.Series(header_series))

    if len(series_list) == 0:
        return None

    return series_list


def event_mode_header_to_WCS(hdr: fits.Header) -> WCS:
    """
    Build a WCS coordinate system from keywords in the header of an event-mode image
    """
    wcs_obj = WCS(naxis=2)
    wcs_obj.wcs.crpix = [hdr["TCRPX6"], hdr["TCRPX7"]]
    wcs_obj.wcs.cdelt = [hdr["TCDLT6"], hdr["TCDLT7"]]
    wcs_obj.wcs.crval = [hdr["TCRVL6"], hdr["TCRVL7"]]
    wcs_obj.wcs.ctype = [hdr["TCTYP6"], hdr["TCTYP7"]]
    return wcs_obj


def _level_2_event_mode_observation_to_series(
    obs: SwiftLevel2FITSObservation,
) -> list[pd.Series] | None:
    """
    For the given event mode observation, pull all SwiftFITSHeaderKeyword keys from the correct header
    and fill in columns of a Series with the same names as the keys
    """

    event_mode_bintable_extension_id = 1

    with fits.open(obs.fits_path) as hdul:
        # only the first extension, which should be a BinTable for event mode images
        hdu = hdul[event_mode_bintable_extension_id]

        if not is_event_mode_bintable(hdu=hdu):
            print(f"Skipping {obs.fits_path}: not a BinTableHDU at extension 1")
            return None

        hdr: fits.Header = hdu.header  # type: ignore
        header_series = {
            k: hdr.get(k, None) for k in _level_2_fits_header_keys_to_extract
        }
        header_series["WCS"] = event_mode_header_to_WCS(hdr)
        header_series["EXTENSION"] = event_mode_bintable_extension_id
        header_series["FITS_FILENAME"] = str(obs.fits_path.name)
        header_series["FULL_FITS_PATH"] = obs.fits_path

    return [pd.Series(header_series)]


def level_2_observation_to_series(
    obs: SwiftLevel2FITSObservation,
) -> list[pd.Series] | None:

    if obs.observation_mode == UvotImageMode.data_mode:
        return _level_2_data_mode_fits_header_to_columns(obs=obs)
    elif obs.observation_mode == UvotImageMode.event_mode:
        return _level_2_event_mode_observation_to_series(obs=obs)
