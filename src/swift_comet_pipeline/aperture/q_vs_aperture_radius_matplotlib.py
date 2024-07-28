import matplotlib.pyplot as plt

from swift_comet_pipeline.aperture.plateau import ProductionPlateau
from swift_comet_pipeline.aperture.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
)


def show_q_vs_aperture_with_plateaus(
    q_vs_aperture_radius_list: list[QvsApertureRadiusEntry],
    q_plateau_list: list[ProductionPlateau] | None,
    km_per_pix: float,
) -> None:

    # TODO: this should take the dataframe, pull out the plateaus from the attrs, and ridgeplot only the rednesses with plateaus

    assert len(set([x.dust_redness for x in q_vs_aperture_radius_list])) == 1
    dust_redness = q_vs_aperture_radius_list[0].dust_redness

    rs_km = [x.aperture_r_km for x in q_vs_aperture_radius_list]
    counts_uw1 = [x.counts_uw1 for x in q_vs_aperture_radius_list]
    counts_uw1_err = [x.counts_uw1_err for x in q_vs_aperture_radius_list]
    counts_uvv = [x.counts_uvv for x in q_vs_aperture_radius_list]
    counts_uvv_err = [x.counts_uvv_err for x in q_vs_aperture_radius_list]
    q_h2os = [x.q_H2O for x in q_vs_aperture_radius_list]
    q_h2os_err = [x.q_H2O_err for x in q_vs_aperture_radius_list]

    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.suptitle(f"dust redness {dust_redness}%/100 nm")

    axs[0].plot(rs_km, counts_uw1, color="#82787f")
    axs[0].errorbar(rs_km, counts_uw1, counts_uw1_err)
    axs[0].set_title(f"uw1 counts vs aperture radius")

    axs[1].plot(rs_km, counts_uvv, color="#82787f")
    axs[1].errorbar(rs_km, counts_uvv, counts_uvv_err)
    axs[1].set_title(f"uvv counts vs aperture radius")

    axs[2].set_yscale("log")
    axs[2].plot(rs_km, q_h2os, color="#a4b7be")
    axs[2].errorbar(rs_km, q_h2os, q_h2os_err)
    axs[2].set_title(f"water production vs aperture radius")

    if q_plateau_list is not None:
        for p in q_plateau_list:
            axs[0].axvspan(
                p.begin_r * km_per_pix,
                p.end_r * km_per_pix,
                color="#afac7c",
                alpha=0.1,
            )
            axs[1].axvspan(
                p.begin_r * km_per_pix,
                p.end_r * km_per_pix,
                color="#afac7c",
                alpha=0.1,
            )
            axs[2].axvspan(
                p.begin_r * km_per_pix,
                p.end_r * km_per_pix,
                color="#afac7c",
                alpha=0.1,
            )
            axs[2].text(
                x=p.begin_r * km_per_pix,
                y=p.end_q / 100,
                s=f"{p.begin_q:1.2e}",
                color="#688894",
                alpha=0.8,
            )
            axs[2].text(
                x=p.end_r * km_per_pix,
                y=p.end_q / 10,
                s=f"{p.end_q:1.2e}",
                color="#688894",
                alpha=0.8,
            )

    plt.show()
