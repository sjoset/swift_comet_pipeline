import numpy as np
from astropy import modeling
from astropy.modeling.models import Gaussian1D
import matplotlib.pyplot as plt

from swift_comet_pipeline.types.comet_profile import CometProfile


# No longer used anywhere - deprecate
def fit_comet_profile_gaussian(comet_profile: CometProfile):
    """Takes a CometRadialProfile and returns a function f(r) generated from the profile's best-fit gaussian"""

    if not comet_profile.center_is_comet_peak:
        print("This function is intended for profiles centered on the comet peak!")
        return None

    # TODO: LevMarLSQFitter is no longer recommended - update if this code is not removed
    lsqf = modeling.fitting.LevMarLSQFitter()
    gaussian = modeling.models.Gaussian1D()
    fitted_gaussian = lsqf(
        gaussian, comet_profile.profile_axis_xs, comet_profile.pixel_values
    )

    return fitted_gaussian


# No longer used anywhere - deprecate
def plot_fitted_gaussian_profile(
    comet_profile: CometProfile,
    fitted_model: Gaussian1D,
    sigma_threshold: float,
    plot_title: str,
) -> None:
    if fitted_model.stddev.value is None:
        print("Fitted gaussian has no information about standard deviation!")
    else:
        plt.vlines(
            x=[
                -sigma_threshold * fitted_model.stddev.value,
                sigma_threshold * fitted_model.stddev.value,
            ],
            ymin=0,
            ymax=np.max(comet_profile.pixel_values),
            color="r",
        )
    plt.scatter(comet_profile.profile_axis_xs, comet_profile.pixel_values)
    plt.plot(
        comet_profile.profile_axis_xs,
        fitted_model(comet_profile.profile_axis_xs),
    )
    plt.title(plot_title)
    plt.show()
