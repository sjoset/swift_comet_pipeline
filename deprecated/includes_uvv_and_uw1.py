from swift_comet_pipeline.scp_types.primitive import *


def includes_uvv_and_uw1_filters(
    obs_log: SwiftUvotObservationLogDataframe,
) -> bool:
    has_uvv_filter = obs_log[obs_log["FILTER"] == UvotFilter.uvv]
    has_uvv_set = set(has_uvv_filter["ORBIT_ID"])

    has_uw1_filter = obs_log[obs_log["FILTER"] == UvotFilter.uw1]
    has_uw1_set = set(has_uw1_filter["ORBIT_ID"])

    has_both = len(has_uvv_set) > 0 and len(has_uw1_set) > 0

    return has_both
