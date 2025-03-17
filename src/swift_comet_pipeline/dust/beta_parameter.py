from swift_comet_pipeline.dust import reddening_correction
from swift_comet_pipeline.pipeline.internal_config.pipeline_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.types import DustReddeningPercent, SwiftFilter


def beta_parameter(
    dust_redness: DustReddeningPercent,
    # solar_spectrum_time: Time | None = None
) -> float:

    from swift_comet_pipeline.spectrum.solar_count_rate import (
        solar_count_rate_in_filter_1au,
    )

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    solar_count_rate_in_uw1 = solar_count_rate_in_filter_1au(
        filter_type=SwiftFilter.uw1
    )
    solar_count_rate_in_uvv = solar_count_rate_in_filter_1au(
        filter_type=SwiftFilter.uvv
    )

    beta_pre_reddening = solar_count_rate_in_uw1.value / solar_count_rate_in_uvv.value
    beta = reddening_correction(dust_redness=dust_redness) * beta_pre_reddening

    return beta
