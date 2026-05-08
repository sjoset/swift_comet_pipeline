from dataclasses import fields
from itertools import product

import pandas as pd

from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter
from swift_comet_pipeline.swift.filters.uvot_filter_to_string import (
    filter_to_file_string,
)


def add_epoch_index_entry_to_dataframe(
    df: pd.DataFrame,
    eid: EpochIndexEntry,
    include_exposure_times: list[UvotFilter] | None = None,
) -> pd.DataFrame:
    """
    Take the given dataframe and add the epoch information in eid as columns, with optional exposure times
    given by the filter list in 'include_exposure_times'.
    """

    if include_exposure_times is None:
        include_exposure_times = []
    dfc = df.copy()

    ex_times = ["exposure_times", "exposure_times_no_veto"]
    # loop over all entries that are not exposure times
    eid_entries = [f.name if f.name not in ex_times else None for f in fields(eid)]
    eid_entries = list(filter(None, eid_entries))

    # grab all of the non-exposure time entries and create a column based on the variable name
    for ee in eid_entries:
        dfc[ee] = getattr(eid, ee)

    # for each filter to be included, add the exposure time before and after veto
    for filter_type, ex_name in product(include_exposure_times, ex_times):
        col_name = ex_name + "_" + filter_to_file_string(filter_type=filter_type)
        filter_dict = getattr(eid, ex_name)
        exp_time = filter_dict.get(filter_type, None)
        if exp_time is not None:
            dfc[col_name] = exp_time

    return dfc
