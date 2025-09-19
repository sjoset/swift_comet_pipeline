import pathlib
from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive import *


@dataclass
class SwiftLevel2FITSObservation:
    """
    Describes a FITS file in the data set: its observation & orbit ids, path, filter, and imaging mode

    This could represent multiple images through multiple image extensions in the FITS file
    """

    orbit_id: SwiftOrbitID
    observation_id: SwiftObservationID
    fits_path: pathlib.Path
    observation_mode: UvotImageMode
    filter_type: UvotFilter
