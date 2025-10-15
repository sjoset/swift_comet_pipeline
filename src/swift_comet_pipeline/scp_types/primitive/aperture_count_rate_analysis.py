from dataclasses import dataclass, fields


@dataclass
class ApertureCountRateAnalysis:
    """
    Given an aperture, we want these results for determining the signal within it
    """

    sum_count_rate: float
    median_count_rate: float
    mean_count_rate: float
    # calculated with the sum of the count rates over the exposure time
    count_rate_shot_noise_variance: float
    # what variance the background contributed, total
    bg_variance: float
    # total variance of the sum: count_rate_shot_noise_variance + bg_variance
    sum_count_rate_variance: float
    # variance if we use the median
    # count_rate_shot_noise_variance * pi/2 + bg_variance

    # if any sigma clipping was used
    sigma_clip: float
    # total valid pixels left in aperture for signal - sigma clipping may remove some
    ap_num_pixels: float


# asdict() deep copies all the values so we do this instead
def aperture_count_rate_analysis_kwargs(acra: ApertureCountRateAnalysis) -> dict:
    return {f.name: getattr(acra, f.name) for f in fields(ApertureCountRateAnalysis)}
