import pathlib
from dataclasses import dataclass

from astropy.time import Time


from swift_comet_pipeline.scp_types.primitive import *


# this should mirror the observation_log_schema() after being loaded and properly type-coerced
@dataclass
class SwiftUvotObservationLogEntry:
    observation_id: SwiftObservationID
    observation_start: Time
    observation_end: Time
    observation_mid: Time
    filter_type: UvotFilter
    comet_fits_ra_deg: float
    comet_fits_dec_deg: float
    comet_ra_rate_arcsec_per_min: float
    comet_dec_rate_arcsec_per_min: float
    comet_sky_motion_arcsec_per_min: float
    comet_sky_motion_position_angle_deg: float
    ion_tail_position_angle_deg: float
    exposure_time_s: float
    fits_extension: int
    fits_filename: str
    fits_full_path: pathlib.Path
    rh_au: float
    rh_rate_km_s: float
    delta_au: float
    phase_angle_deg: float
    comet_horizons_ra_deg: float
    comet_horizons_dec_deg: float
    comet_pixel_x: float
    comet_pixel_y: float
    user_comet_pixel_x: float
    user_comet_pixel_y: float
    uvot_datamode: UvotImageMode
    arcsecs_per_pixel: UvotPixelResolution
    km_per_pix: float
    fits_creator: str
    manual_veto: bool
    epoch_id: EpochID


SwiftUvotObservationLog: TypeAlias = list[SwiftUvotObservationLogEntry]
