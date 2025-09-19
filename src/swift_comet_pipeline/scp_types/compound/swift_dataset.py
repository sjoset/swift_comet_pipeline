from typing import Protocol

from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.compound.swift_level_2_fits import (
    SwiftLevel2FITSObservation,
)


class SwiftDataset(Protocol):
    orbit_ids: list[SwiftOrbitID] | None
    observation_ids: list[SwiftObservationID] | None
    observations: dict[
        tuple[SwiftObservationID, UvotFilter], list[SwiftLevel2FITSObservation] | None
    ]

    def get_swift_uvot_event_mode_fits_observations(
        self, obsid: SwiftObservationID, filter_type: UvotFilter
    ) -> list[SwiftLevel2FITSObservation] | None: ...

    def get_swift_uvot_data_mode_fits_observations(
        self,
        obsid: SwiftObservationID,
        filter_type: UvotFilter,
        image_type: SwiftUvotImageType = SwiftUvotImageType.sky_units,
    ) -> list[SwiftLevel2FITSObservation] | None: ...

    def get_swift_uvot_observations(
        self, obsid: SwiftObservationID, filter_type: UvotFilter
    ) -> list[SwiftLevel2FITSObservation] | None: ...

    def get_observation_image_directory(
        self, obsid: SwiftObservationID, image_mode: UvotImageMode
    ) -> pathlib.Path: ...

    def get_observation_image(
        self,
        obsid: SwiftObservationID,
        image_mode: UvotImageMode,
        fits_filename: str,
        extension_id: int,
    ) -> SwiftUvotImage | None: ...
