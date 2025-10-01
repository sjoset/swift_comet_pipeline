import pyarrow as pa

from swift_comet_pipeline.scp_types.primitive import *


def observation_log_schema() -> pa.Schema:
    """
    Returns schema that describes the columns in our observation log dataframe when written to/read from disk

    Most FITS header keyword documentation can be found at https://archive.stsci.edu/swiftuvot/UVOT_swguide_v2_2.pdf
    """
    schema_list = [
        ### FITS keywords from SWIFT-processed FITS files
        # Observation ID: uniquely identifies an SWIFT observation set, which may contain multiple FITS image extensions.  Represented as a fixed-width string of 11 characters,
        # with the first 8 digits being the 'target id' and the last 3 being 'segment number': '000', '001', ... to denote multiple observations taken on the target consecutively.
        # We store this in the db as an integer, and convert back to SwiftObservationID string representation when we read the file
        pa.field("OBS_ID", pa.int64()),
        # Start time of the observation, as a string with the format 'YYYY-MM-DD HH:MM:SS'
        pa.field("DATE_OBS", pa.string()),
        # End time of the observation, as a string with the format 'YYYY-MM-DD HH:MM:SS'
        pa.field("DATE_END", pa.string()),
        # Mid time of the observation, as a string with the format 'YYYY-MM-DD HH:MM:SS'
        pa.field("MID_TIME", pa.string()),
        # Which UvotFilter was used for this observation
        pa.field("FILTER", pa.string()),
        # Right ascension, degrees, according to source FITS file
        pa.field("RA_OBJ", pa.float64()),
        # Declination, degrees, according to source FITS file
        pa.field("DEC_OBJ", pa.float64()),
        # Rate of change of RA in arcsec/min
        pa.field("RA_RATE", pa.float64()),
        # Rate of change of Dec in arcsec/min
        pa.field("DEC_RATE", pa.float64()),
        # Total rate of sky motion in arcsec/min
        pa.field("SKY_MOTION", pa.float64()),
        # Direction of sky motion, position angle, degrees (dust tail is 180 degrees opposite to this)
        pa.field("SKY_MOTION_PA", pa.float64()),
        # Position angle of ion tail, degrees
        pa.field("ION_TAIL_PA", pa.float64()),
        # Total exposure time after all known corrections applied
        pa.field("EXPOSURE", pa.float64()),
        # Each image extension gets its own header and entry in the observation log: keep track of it here
        pa.field("EXTENSION", pa.int16()),
        # The filename (not path) of the FITS file this observation came from: together with the extension, we can find an observation's image to read and process later
        pa.field("FITS_FILENAME", pa.string()),
        # Full path to source Swift FITS file
        pa.field("FULL_FITS_PATH", pa.string()),
        # heliocentric distance at time of observation, in AU
        pa.field("HELIO", pa.float64()),
        # heliocentric velocity at time of observation, in km/s
        pa.field("HELIO_V", pa.float64()),
        # distance to comet, AU
        pa.field("OBS_DIS", pa.float64()),
        # phase angle, degrees (Sun-Target-Object angle)
        pa.field("PHASE", pa.float64()),
        # coordinates given by Horizons for the center of the comet
        pa.field("RA", pa.float64()),
        pa.field("DEC", pa.float64()),
        # PX, PY: pixel x and y coordinates of comet center after converting RA, DEC using WCS
        pa.field("PX", pa.float64()),
        pa.field("PY", pa.float64()),
        # x and y coordinates of comet center as selected by user during veto step
        pa.field("USER_CENTER_X", pa.float64()),
        pa.field("USER_CENTER_Y", pa.float64()),
        # string as taken from the FITS header keyword DATAMODE
        pa.field("DATAMODE", pa.string()),
        # how many arcseconds per pixel in this data mode
        pa.field("ARCSECS_PER_PIXEL", pa.float64()),
        # conversion from pixels to kilometers, based on the distance of the comet from the observation point
        pa.field("KM_PER_PIX", pa.float64()),
        # String holding the SWIFT software that processed the image, which includes the version.  Currently unused - this is here just in case.
        pa.field("CREATOR", pa.string()),
        # Holds whether or not the user has chosen to veto the image's inclusion in stacking: manual_veto = true means exclude
        pa.field("manual_veto", pa.bool_()),
        # Holds the epoch ID after user has time-sliced the observations into epochs
        pa.field("epoch_id", pa.string()),
    ]

    schema = pa.schema(schema_list)

    return schema
