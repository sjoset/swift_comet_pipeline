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


def show_q_density_estimates_vs_redness(
    q_vs_aperture_radius_list: list[QvsApertureRadiusEntry],
    q_plateau_list_dict: ReddeningToProductionPlateauListDict,
) -> None:
    """
    Takes the results in list q_vs_aperture_radius_list and computes a histogram of Q(h2o) values that appear at each dust_redness, plots
    each histogram in its own row, and highlights the production values where a plateau was detected
    """

    # take log10 of the water production and filter any negative production values
    full_df = dataframe_from_q_vs_aperture_radius_entry_list(
        q_vs_r=q_vs_aperture_radius_list
    )
    full_df["log_q"] = full_df["q_H2O"].apply(np.log10)
    df_mask = full_df["log_q"] > 0

    df = full_df[df_mask]

    num_dust_rednesses = len(set(df["dust_redness"]))
    min_redness = np.min(df["dust_redness"])

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    graph_q_min = np.min(df.log_q) - 0.2
    graph_q_max = np.max(df.log_q) + 0.1

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
        row_order=sorted(list(set(df.dust_redness)), reverse=True),
        sharex=True,
        xlim=(graph_q_min, graph_q_max),
        sharey=True,
    )

    g.map(
        sns.kdeplot,
        "log_q",
        bw_adjust=0.25,
        # TODO: this clips the histogram, but not the axes
        # clip_on=False,
        clip=(26, 30),
        fill=True,
        alpha=0.8,
        linewidth=0.5,
    )
    g.map(sns.kdeplot, "log_q", clip_on=False, color="w", lw=1.0, bw_adjust=0.25)
    g.refline(y=0, linewidth=1.0, linestyle="-", color=None, clip_on=False)  # type: ignore

    g.set(xlim=(26, 30))
    # plt.xlim(26, 30)

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

    # maps the 'plateau_quality' range to a color - arbitrarily chosen after poking around with some data
    plateau_color_map = {"#82787f", "#afac7c", "#e7e7ea"}
    plateau_thresholds = [(0.0, 0.3), (0.3, 0.6), (0.6, 100.0)]

    def plateau_spans(dust_redness_df, color, label):
        """
        For the given subplot, draw vertical bars mapping where plateaus in production were found
        """
        # we ignore both of these, but we can't use the _ to ignore both of them: this causes an error
        # so we del them immediately after the function call and the linter stops complaining about unused vars
        del dust_redness_df, color
        dust_redness = DustReddeningPercent(float(label))
        if len(q_plateau_list_dict[dust_redness]) == 0:
            return
        for qpld in q_plateau_list_dict[dust_redness]:
            ax = plt.gca()
            average_r = (qpld.end_r + qpld.begin_r) / 2
            plateau_quality = np.abs((qpld.end_r**2 - qpld.begin_r**2) / average_r**2)
            for pt, pcolor in zip(plateau_thresholds, plateau_color_map):
                if plateau_quality > pt[0] and plateau_quality < pt[1]:
                    plateau_color = pcolor

            ax.axvspan(
                np.log10(qpld.begin_q),
                np.log10(qpld.end_q),
                color=plateau_color,
                alpha=0.3,
                ymax=0.5,
            )

    g.map(plateau_spans, "dust_redness")

    g.figure.subplots_adjust(hspace=-0.25)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set(
        xticks=list(
            np.linspace(
                np.floor(graph_q_min),
                np.ceil(graph_q_max),
                num=(np.ceil(graph_q_max) - np.floor(graph_q_min) + 1).astype(np.int32),
                endpoint=True,
            )
        )
    )
    g.despine(bottom=True, left=True)

    fig = plt.gcf()
    fig.suptitle("q histogram vs dust redness")
    # plt.xlim(24, np.ceil(graph_q_max))
    plt.show()
