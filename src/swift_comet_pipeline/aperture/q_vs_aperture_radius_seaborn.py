import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from swift_comet_pipeline.aperture.q_vs_aperture_radius import (
    ReddeningToProductionPlateauListDict,
)
from swift_comet_pipeline.aperture.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
    dataframe_from_q_vs_aperture_radius_entry_list,
)


def show_q_vs_aperture_radius_seaborn(
    q_vs_aperture_radius_list: list[QvsApertureRadiusEntry],
    q_plateau_list_dict: ReddeningToProductionPlateauListDict,
    km_per_pix: float,
) -> None:
    """
    Shows a scatterplot of log(q_h2o) vs aperture radius in km, with all data combined and colored according to the
    assumed dust color.  Also indicates plateau concentrations by drawing a vspan at a low alpha value where a plateau
    occurs, leading to shades where the plateaus from different rednesses agree.
    """

    full_df = dataframe_from_q_vs_aperture_radius_entry_list(
        q_vs_r=q_vs_aperture_radius_list
    )
    full_df["log_q"] = full_df["q_H2O"].apply(np.log10)
    df_mask = full_df["log_q"] > 0

    df = full_df[df_mask]

    dust_rednesses = list(set(df.dust_redness))
    num_dust_rednesses = len(dust_rednesses)
    min_redness = np.min(df.dust_redness)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(
        num_dust_rednesses,
        rot=0.0,
        start=1.0,
        dark=float(min_redness / 100),
        light=0.7,
        reverse=True,
        hue=1.0,
    )

    ax = plt.gca()

    for dust_redness in dust_rednesses:
        q_plateau_list = q_plateau_list_dict[dust_redness]
        if len(q_plateau_list) == 0:
            continue

        for p in q_plateau_list:
            ax.axvspan(
                p.begin_r * km_per_pix,
                p.end_r * km_per_pix,
                color="#e7e7ea",
                alpha=0.05,
            )

    sns.scatterplot(
        data=df,  # type: ignore
        x="aperture_r_km",
        y="log_q",
        hue="dust_redness",
        palette=pal,
        legend=False,
    )

    plt.show()
