from swift_comet_pipeline.scp_types.primitive import *


def swift_observation_id_from_int(number: int) -> SwiftObservationID | None:
    converted_string = f"{number:011}"
    if len(converted_string) != 11:
        return None
    return SwiftObservationID(converted_string)
