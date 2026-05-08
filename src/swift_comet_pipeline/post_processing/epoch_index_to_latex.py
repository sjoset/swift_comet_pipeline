import pandas as pd

from swift_comet_pipeline.scp_types.compound import EpochIndex
from swift_comet_pipeline.scp_types.primitive import *


def exposure_time_records(row: pd.Series) -> list[dict]:

    pre_veto_exposure = row.exposure_times_no_veto
    post_veto_exposure = row.exposure_times

    if set(pre_veto_exposure.keys()) != set(post_veto_exposure.keys()):
        print("Pre and post veto filters do not match!")
        exit(1)
    filter_types = pre_veto_exposure.keys()

    return [
        {
            "filter": filter_type,
            "pre_veto_t": pre_veto_exposure[filter_type],
            "post_veto_t": post_veto_exposure[filter_type],
        }
        for filter_type in filter_types
    ]


def epoch_index_to_dataframe(epoch_index: EpochIndex) -> pd.DataFrame:

    df = pd.DataFrame(data=epoch_index)

    # turn the exposure time column from a dictionary of {filter -> exposure time} into a bunch of separate rows
    exploded_df = df.assign(exposures=df.apply(exposure_time_records, axis=1)).explode(
        "exposures"
    )
    exploded_df[
        ["filter_type", "exposure_time_pre_veto", "exposure_time_post_veto"]
    ] = pd.DataFrame(exploded_df["exposures"].to_list(), index=exploded_df.index)
    exploded_df = exploded_df.drop(
        columns=["exposures", "exposure_times", "exposure_times_no_veto"]
    ).reset_index(drop=True)

    return exploded_df


def epoch_index_to_latex(epoch_index: EpochIndex) -> str:

    # TODO: write formatters for each column to touch up the output
    return epoch_index_to_dataframe(epoch_index=epoch_index).to_latex()
