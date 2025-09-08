from swift_comet_pipeline.types import DustReddeningPercent


# TODO: cite papers that justify this mean and sigma for assumed comet dust redness


# Assume a gaussian distribution of comet dust redness, with mean and standard deviation below
def get_dust_redness_mean_prior() -> DustReddeningPercent:
    """
    TODO: fill in DOI below
    Alvarez-Candal 2025, "X-SHOOTER Spectrum of Comet C/2025 N1: Insights into a Distant Interstellar Visitor", doi:

    TODO: place citations
    """
    return DustReddeningPercent(35.0)


def get_dust_redness_sigma_prior() -> DustReddeningPercent:
    return DustReddeningPercent(10.0)
