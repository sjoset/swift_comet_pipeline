from functools import cache

from swift_comet_pipeline.modeling.spectra.solar.solar_count_rate import (
    solar_count_rate_in_filter_1au,
)
from swift_comet_pipeline.photometry.dust.reddening_correction import (
    reddening_correction,
)
from swift_comet_pipeline.pipeline.internal_config.pipeline_internal_config import (
    read_swift_pipeline_config,
)
from swift_comet_pipeline.scp_types.primitive import *


@cache
def beta_parameter(dust_redness: DustReddeningPercent) -> float:

    # from swift_comet_pipeline.spectrum.solar_count_rate import (
    #     solar_count_rate_in_filter_1au,
    # )

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not read pipeline configuration!")
        exit(1)

    solar_count_rate_in_uw1 = solar_count_rate_in_filter_1au(filter_type=UvotFilter.uw1)
    solar_count_rate_in_uvv = solar_count_rate_in_filter_1au(filter_type=UvotFilter.uvv)

    beta_pre_reddening = solar_count_rate_in_uw1.value / solar_count_rate_in_uvv.value
    beta = reddening_correction(dust_redness=dust_redness) * beta_pre_reddening

    return beta
