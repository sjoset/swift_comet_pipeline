from swift_comet_pipeline.scp_types.primitive import *


def swift_orbit_id_from_obsid(obsid: SwiftObservationID) -> SwiftOrbitID:
    obsid_int = int(obsid)
    orbit_int = round(obsid_int / 1000)
    return SwiftOrbitID(f"{orbit_int:08}")


def swift_orbit_id_from_int(number: int) -> SwiftOrbitID | None:
    converted_string = f"{number:08}"
    if len(converted_string) != 8:
        return None
    return SwiftOrbitID(converted_string)
