import logging as log


from swift_comet_pipeline.scp_types.compound.epoch_index import (
    EpochIndex,
    EpochIndexEntry,
)
from swift_comet_pipeline.scp_types.primitive.epoch_id import EpochID


def get_epoch_index_entry(
    epoch_index: EpochIndex, epoch_id: EpochID
) -> EpochIndexEntry | None:

    matching_entries = [x for x in epoch_index if x.epoch_id == epoch_id]

    if len(matching_entries) != 1:
        log.warn(
            f"Non-unique epoch id found while looking through epoch index for {epoch_id}: found {len(matching_entries)}!"
        )
        return None

    if len(matching_entries) == 0:
        log.info(f"No matching entries in epoch index for {epoch_id}!")
        return None

    return matching_entries[0]
