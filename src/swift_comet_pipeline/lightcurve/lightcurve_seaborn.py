import numpy as np

# import seaborn as sns
# import seaborn.objects as so
# import astropy.units as u

from swift_comet_pipeline.lightcurve.lightcurve import LightCurve


def show_lightcurve_seaborn(lc: LightCurve, best_lc: LightCurve | None = None) -> None:

    # because I don't understand seaborn's axis scaling with objects
    # df.q = np.log10(df.q)
    # df.q_err = np.log10(df.q_err)
    # df.q_lower_bound = np.log10(df.q_lower_bound)
    # df.q_upper_bound = np.log10(df.q_upper_bound)

    # sns.set_theme()

    # p1 = so.Plot(
    #     data=df,
    #     x="rh",
    #     y="q",
    #     # ymin="q_lower_bound",
    #     # ymax="q_upper_bound",
    #     color="dust_redness",
    # ).add(so.Dot(alpha=0.2), so.Dodge(), so.Jitter(0.5))
    #
    # p1.show()

    pass
