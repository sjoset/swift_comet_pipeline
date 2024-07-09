import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# import seaborn as sns
# import seaborn.objects as so
import astropy.units as u

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import (
    LightCurve,
    lightcurve_to_dataframe,
)


def show_lightcurve(lc: LightCurve, best_lc: LightCurve | None = None) -> None:

    lc_cleaned: LightCurve = [x for x in lc if x is not None]
    df_raw = lightcurve_to_dataframe(lc=lc_cleaned)

    df_raw["rh"] = df_raw.rh_au * np.sign(df_raw.time_from_perihelion_days)
    df_raw["q_upper_bound"] = df_raw.q + df_raw.q_err / 2
    df_raw["q_lower_bound"] = df_raw.q - df_raw.q_err / 2

    # add jitter to x-values
    df_raw.time_from_perihelion_days = (
        df_raw.time_from_perihelion_days
        + np.random.uniform(low=-10, high=10, size=len(df_raw))
    )
    df_raw.rh = df_raw.rh + np.random.uniform(low=-0.15, high=0.15, size=len(df_raw))

    positive_production_mask = np.logical_and(
        df_raw.q > 0.0, df_raw.q_lower_bound > 0.0
    )
    df = df_raw[positive_production_mask].copy()

    dust_rednesses = list(set(df.dust_redness))
    dust_cmap = LinearSegmentedColormap.from_list(
        name="custom", colors=["#8e8e8e", "#c74a77"], N=(len(dust_rednesses) + 1)
    )

    _, ax = plt.subplots()

    ax.scatter(
        df.time_from_perihelion_days,
        df.q,
        c=df.dust_redness,
        cmap=dust_cmap,
        alpha=0.2,
    )
    plt.errorbar(
        df.time_from_perihelion_days, df.q, yerr=df.q_err, alpha=0.25, ls="none"
    )

    if best_lc is not None:
        best_lc_cleaned: LightCurve = [x for x in best_lc if x is not None]
        best_df = lightcurve_to_dataframe(lc=best_lc_cleaned)
        best_df["rh"] = best_df.rh_au * np.sign(best_df.time_from_perihelion_days)
        ax.scatter(
            best_df.time_from_perihelion_days,
            best_df.q,
            color="black",
            alpha=1.0,
            s=4,
        )

    plt.yscale("log")
    plt.show()


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
