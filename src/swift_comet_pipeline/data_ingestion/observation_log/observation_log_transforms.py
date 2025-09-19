import pandas as pd

from swift_comet_pipeline.scp_types.compound.swift_uvot_observation_log import (
    SwiftUvotObservationLog,
    SwiftUvotObservationLogEntry,
)
from swift_comet_pipeline.scp_types.primitive.swift_uvot_observation_log_dataframe import (
    SwiftUvotObservationLogDataframe,
)


# this should be in sync with swift_uvot_observation_log and observation_log_schema()
_dataframe_column_to_field_name = {
    "OBS_ID": "observation_id",
    "DATE_OBS": "observation_start",
    "DATE_END": "observation_end",
    "MID_TIME": "observation_mid",
    "FILTER": "filter_type",
    "RA_OBJ": "comet_fits_ra_deg",
    "DEC_OBJ": "comet_fits_dec_deg",
    "RA_RATE": "comet_ra_rate_arcsec_per_min",
    "DEC_RATE": "comet_dec_rate_arcsec_per_min",
    "SKY_MOTION": "comet_sky_motion_arcsec_per_min",
    "SKY_MOTION_PA": "comet_sky_motion_position_angle_deg",
    "ION_TAIL_PA": "ion_tail_position_angle_deg",
    "EXPOSURE": "exposure_time_s",
    "EXTENSION": "fits_extension",
    "FITS_FILENAME": "fits_filename",
    "FULL_FITS_PATH": "fits_full_path",
    "HELIO": "rh_au",
    "HELIO_V": "rh_rate_km_s",
    "OBS_DIS": "delta_au",
    "PHASE": "phase_angle_deg",
    "RA": "comet_horizons_ra_deg",
    "DEC": "comet_horizons_dec_deg",
    "PX": "comet_pixel_x",
    "PY": "comet_pixel_y",
    "USER_CENTER_X": "user_comet_pixel_x",
    "USER_CENTER_Y": "user_comet_pixel_y",
    "DATAMODE": "uvot_datamode",
    "ARCSECS_PER_PIXEL": "arcsecs_per_pixel",
    "KM_PER_PIX": "km_per_pix",
    "CREATOR": "fits_creator",
    "manual_veto": "manual_veto",
    "epoch_id": "epoch_id",
}


def swift_uvot_observation_log_dataframe_to_entry_list(
    obs_log_df: SwiftUvotObservationLogDataframe,
) -> SwiftUvotObservationLog:

    entry_list = []
    for rec in obs_log_df.to_dict(orient="records"):

        entry_dict = {}
        for column_name, field_name in _dataframe_column_to_field_name.items():
            if column_name not in rec:
                continue
            entry_dict[field_name] = obs_log_df[column_name]

        entry_list.append(SwiftUvotObservationLogEntry(**entry_dict))

    return entry_list


def swift_uvot_observation_log_to_dataframe(
    obs_log: SwiftUvotObservationLog,
) -> SwiftUvotObservationLogDataframe:

    df_records = []
    # loop through each observation log entry
    for entry in obs_log:

        # and build a dictionary that defines this row
        entry_dict = {}
        for column_name, field_name in _dataframe_column_to_field_name:
            entry_dict[column_name] = getattr(entry, field_name)

        # and append it to our rows
        df_records.append(entry_dict)

    obs_log_df = pd.DataFrame.from_records(
        data=df_records, columns=_dataframe_column_to_field_name.keys()
    )
    return obs_log_df
