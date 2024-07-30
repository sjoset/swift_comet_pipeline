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
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent


def show_plateau_distribution_seaborn(
    q_vs_aperture_radius_list: list[QvsApertureRadiusEntry],
    q_plateau_list_dict: ReddeningToProductionPlateauListDict,
    km_per_pix: float,
) -> None:

    full_df = dataframe_from_q_vs_aperture_radius_entry_list(
        q_vs_r=q_vs_aperture_radius_list
    )
    full_df["log_q"] = full_df["q_H2O"].apply(np.log10)
    df_mask = full_df["log_q"] > 0

    df = full_df[df_mask]

    num_dust_rednesses = len(set(df["dust_redness"]))
    min_redness = np.min(df["dust_redness"])

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    graph_r_km_min = np.min(df.aperture_r_km)
    graph_r_km_max = np.max(df.aperture_r_km)

    pal = sns.cubehelix_palette(
        num_dust_rednesses,
        rot=0.0,
        start=1.0,
        dark=float(min_redness / 100),
        light=0.7,
        reverse=True,
        hue=1.0,
    )
    g = sns.FacetGrid(
        data=df,  # type: ignore
        row="dust_redness",
        hue="dust_redness",
        aspect=10,
        height=0.5,
        palette=pal,
        row_order=sorted(list(set(df.dust_redness)), reverse=True),  # type: ignore
        sharex=True,
        xlim=(graph_r_km_min, graph_r_km_max),
        sharey=True,
    )

    # g.map(
    #     sns.lineplot,
    #     "aperture_r_km",
    #     "log_q",
    #     clip_on=False,
    #     color="b",
    #     alpha=0.5,
    #     lw=1.0,
    # )
    g.refline(y=0, linewidth=1.0, linestyle="-", color=None, clip_on=False)  # type: ignore

    # Define and use a simple function to label the plot in axes coordinates
    def label(_, color, label):
        """
        Put the dust redness labels on the left hand side of the graph
        """
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="normal",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "dust_redness")

    def plateau_spans(_, color, label):
        """
        For the given subplot, draw vertical bars mapping where plateaus in production were found
        """
        dust_redness = DustReddeningPercent(float(label))
        if len(q_plateau_list_dict[dust_redness]) == 0:
            return
        for qpld in q_plateau_list_dict[dust_redness]:
            ax = plt.gca()
            ax.axvspan(
                qpld.begin_r * km_per_pix,
                qpld.end_r * km_per_pix,
                color=color,
                alpha=0.3,
                ymax=0.7,
            )

    g.map(plateau_spans, "dust_redness")

    g.figure.subplots_adjust(hspace=-0.25)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    # g.set(
    #     xticks=list(
    #         np.linspace(
    #             np.floor(graph_q_min),
    #             np.ceil(graph_q_max),
    #             num=(np.ceil(graph_q_max) - np.floor(graph_q_min) + 1).astype(np.int32),
    #             endpoint=True,
    #         )
    #     )
    # )
    g.despine(bottom=True, left=True)

    # fig = plt.gcf()
    # plt.tight_layout()
    plt.show()
