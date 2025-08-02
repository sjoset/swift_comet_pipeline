from astropy.io import fits
from astropy.time import Time


def uvot_time_to_astropy_time(uvot_t: float, event_mode_hdr: fits.Header) -> Time:
    # hdr: fits.header.Header = event_mode_hdr.header
    hdr = event_mode_hdr

    # required FITS timing keywords (OGIP/FITS convention)
    mjdref: float = float(hdr.get("MJDREFI", 0.0)) + float(hdr.get("MJDREFF", 0.0))  # type: ignore
    timezero: float = float(hdr.get("TIMEZERO", 0.0))  # type: ignore
    timeunit: float = hdr.get("TIMEUNIT", "s").strip().lower()  # type: ignore
    timesys: float = hdr.get("TIMESYS", "TT").strip().lower()  # type: ignore

    # If TIME is in days already, factor should be 1 (per OGIP rules).
    # Seconds --> days if needed
    time_factor = 1.0 / 86400.0 if timeunit.startswith("s") else 1.0  # type: ignore

    # Apply any global TIMEZERO offset (OGIP/93-003)
    t_true = uvot_t + timezero

    # Optional: apply Swift clock offset if not already applied (see note below)
    if str(hdr.get("CLOCKAPP", "T")).strip().upper().startswith("F"):
        utcf = hdr.get("UTCFINIT", hdr.get("UTCF", 0.0))  # seconds
        t_true = t_true + float(utcf)  # type: ignore

    # Build MJD in the file’s native time scale (usually TT), then convert to UTC
    mjd_native = mjdref + t_true * time_factor
    t_astropy = Time(mjd_native, format="mjd", scale=timesys)  # e.g., scale='tt'

    return t_astropy
