from swift_comet_pipeline.types import DustReddeningPercent


# TODO: cite papers that justify this mean and sigma for assumed comet dust redness


# Assume a gaussian distribution of comet dust redness, with mean and standard deviation below
def get_dust_redness_mean_prior() -> DustReddeningPercent:
    return DustReddeningPercent(35.0)


def get_dust_redness_sigma_prior() -> DustReddeningPercent:
    return DustReddeningPercent(5.0)
