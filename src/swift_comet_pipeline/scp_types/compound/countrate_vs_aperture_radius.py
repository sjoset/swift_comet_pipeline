from dataclasses import dataclass

from swift_comet_pipeline.scp_types.compound.count_rate import CountRate


@dataclass
class CountrateVsApertureRadius:
    r_pixels: list[float]
    count_rates: list[CountRate]
