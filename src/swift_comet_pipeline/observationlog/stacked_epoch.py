from typing import TypeAlias

from swift_comet_pipeline.observationlog.epoch import Epoch


"""
A StackedEpoch is structurally identical to an Epoch (which itself is an alias of SwiftObservationLog).
However, it should only have non-vetoed data rows from the uw1 and uvv filters that were involved in producing the associated
stacked image for that epoch.  This decouples the filtering logic that determines what images are included in the stack from
the later processing: all of the SWIFT images referred to in a StackedEpoch were included and may be processed without re-checking
if any particular observation should be ignored or included.
"""
StackedEpoch: TypeAlias = Epoch
