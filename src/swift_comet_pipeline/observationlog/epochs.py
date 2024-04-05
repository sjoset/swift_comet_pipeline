import pathlib
import pandas as pd
import pyarrow as pa

from typing import TypeAlias

from swift_comet_pipeline.observationlog.observation_log import (
    read_observation_log,
    write_observation_log,
    SwiftObservationLog,
)


Epoch: TypeAlias = pd.DataFrame

# For saving metadata with a parquet file
# epoch_stack_schema = pa.schema(
#     [],
#     metadata={
#         "coincidence_correction": str(do_coincidence_correction),
#         "pixel_resolution": str(epoch_pixel_resolution),
#     },
# )


# TODO: instead of breaking into smaller epochs, can we just add a column to the observation log with an epoch id, and
# filter by that after loading?
# TODO: merge this schema into observation log and just initialize all these possible error values in build_observation_log
def epoch_schema() -> pa.lib.Schema:
    epoch_schema = pa.schema(
        [
            pa.field("manual_veto", pa.bool_()),
        ]
    )

    return epoch_schema


def read_epoch(epoch_path: pathlib.Path) -> Epoch:
    """
    Allow read_observation_log to do post-load processing on SwiftObservationLog columns
    """
    epoch = read_observation_log(epoch_path, additional_schema=epoch_schema())

    return epoch


def write_epoch(
    epoch: Epoch, epoch_path: pathlib.Path, additional_schema: pa.lib.Schema = None
) -> None:
    schema = epoch_schema()
    if additional_schema is not None:
        schema = pa.unify_schemas([schema, additional_schema])

    # do any column processing of our own here

    write_observation_log(epoch, epoch_path, additional_schema=schema)


def epoch_from_obs_log(obs_log: SwiftObservationLog) -> Epoch:
    """
    Adds default values to columns from an observation log so that it matches epoch_schema
    """
    obs_log["manual_veto"] = False

    return obs_log
