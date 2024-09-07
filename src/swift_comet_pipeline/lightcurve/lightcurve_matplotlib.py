import numpy as np

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from swift_comet_pipeline.lightcurve.lightcurve import (
    LightCurve,
    lightcurve_to_dataframe,
)


def show_lightcurve_mpl(lc: LightCurve, best_lc: LightCurve | None = None) -> None:

    x_column = "rh"
    # x_column = "time_from_perihelion_days"

    # TODO: can probably use np.min(np.diff())/3 for jitter
    # x-axis jitter settings
    time_jitter_days = 2.0
    rh_jitter_au = 0.07

    lc_cleaned: LightCurve = [x for x in lc if x is not None]
    df_raw = lightcurve_to_dataframe(lc=lc_cleaned)

    df_raw["rh"] = df_raw.rh_au * np.sign(df_raw.time_from_perihelion_days)
    df_raw["q_upper_bound"] = df_raw.q + df_raw.q_err / 2
    df_raw["q_lower_bound"] = df_raw.q - df_raw.q_err / 2

    # add jitter to x-values
    df_raw.time_from_perihelion_days = (
        df_raw.time_from_perihelion_days
        + np.random.uniform(
            low=-time_jitter_days, high=time_jitter_days, size=len(df_raw)
        )
    )
    df_raw.rh = df_raw.rh + np.random.uniform(
        low=-rh_jitter_au, high=rh_jitter_au, size=len(df_raw)
    )

    # only take the positive production results for display
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
        df[x_column],
        df.q,
        c=df.dust_redness,
        cmap=dust_cmap,
        alpha=0.2,
    )
    plt.errorbar(df[x_column], df.q, yerr=df.q_err, alpha=0.25, ls="none")

    if best_lc is not None:
        best_lc_cleaned: LightCurve = [x for x in best_lc if x is not None]
        best_df = lightcurve_to_dataframe(lc=best_lc_cleaned)
        best_df["rh"] = best_df.rh_au * np.sign(best_df.time_from_perihelion_days)
        ax.scatter(
            best_df[x_column],
            best_df.q,
            color="black",
            alpha=1.0,
            s=4,
        )

    plt.suptitle("C/2012 K1")
    plt.xlabel(xlabel=f"{x_column} AU")
    plt.ylabel(ylabel="log Q(H2O)")
    plt.yscale("log")
    plt.show()
