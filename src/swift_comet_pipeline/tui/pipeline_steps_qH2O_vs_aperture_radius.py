from itertools import product, groupby
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rich import print as rprint
import astropy.units as u

from swift_comet_pipeline.background.background_result import (
    BackgroundResult,
    dict_to_background_result,
)
from swift_comet_pipeline.comet.comet_aperture import comet_manual_aperture
from swift_comet_pipeline.comet.comet_center_finding import compare_comet_center_methods
from swift_comet_pipeline.comet.plateau_detect import (
    ProductionPlateau,
    dust_plateau_list_dict_serialize,
    find_production_plateaus,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.count_rate import CountRate
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage, get_uvot_image_center
from swift_comet_pipeline.tui.tui_common import stacked_epoch_menu
from swift_comet_pipeline.water_production.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.water_production.flux_OH import (
    OH_flux_from_count_rate,
    beta_parameter,
)
from swift_comet_pipeline.water_production.num_OH_to_Q import num_OH_to_Q_vectorial
from swift_comet_pipeline.water_production.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
    dataframe_from_q_vs_aperture_radius_entry_list,
)


# TODO: make the visualization separate so we can load the csv and look at it later


@dataclass
class CountrateVsApertureRadius:
    r_pixels: list[float]
    count_rates: list[CountRate]


def counts_vs_aperture_radius(
    img: SwiftUVOTImage,
    bg: BackgroundResult,
    r_pixels: list[float],
    use_tqdm: bool = False,
) -> CountrateVsApertureRadius:

    comet_center = get_uvot_image_center(img=img)

    if use_tqdm is True:
        r_pix_list = tqdm(r_pixels)
    else:
        r_pix_list = r_pixels

    count_rates = [
        comet_manual_aperture(
            img=img,
            aperture_center=comet_center,
            aperture_radius=r,
            bg=bg.count_rate_per_pixel,
        )
        for r in r_pix_list
    ]

    return CountrateVsApertureRadius(r_pixels=r_pixels, count_rates=count_rates)


def q_vs_aperture_radius(
    stacked_epoch: Epoch,
    uw1_img: SwiftUVOTImage,
    uvv_img: SwiftUVOTImage,
    dust_rednesses: list[DustReddeningPercent],
    uw1_bg: BackgroundResult,
    uvv_bg: BackgroundResult,
    max_aperture_radius: u.Quantity = 1.5e5 * u.km,  # type: ignore
    num_apertures: int = 300,
) -> list[QvsApertureRadiusEntry] | None:

    helio_r_au = np.mean(stacked_epoch.HELIO)
    helio_v_kms = np.mean(stacked_epoch.HELIO_V)
    delta_au = np.mean(stacked_epoch.OBS_DIS)
    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)

    r_pix_max = max_aperture_radius.to_value(u.km) / km_per_pix  # type: ignore
    if r_pix_max < 1.0:
        print(
            "Physical maximum aperture size given results in aperture size less than a pixel!"
        )
        return None

    aperture_radii_pix = np.linspace(
        start=1,
        stop=np.round(r_pix_max).astype(np.int32),
        num=num_apertures,
        endpoint=True,
    )

    beta_parameters = {x: beta_parameter(x) for x in dust_rednesses}

    print("Calculating uw1 counts ...")
    uw1_count_rates_vs_r = counts_vs_aperture_radius(
        img=uw1_img, bg=uw1_bg, r_pixels=aperture_radii_pix, use_tqdm=True
    )
    print("Calculating uvv counts ...")
    uvv_count_rates_vs_r = counts_vs_aperture_radius(
        img=uvv_img, bg=uvv_bg, r_pixels=aperture_radii_pix, use_tqdm=True
    )

    uw1_magnitudes = [
        magnitude_from_count_rate(count_rate=cr, filter_type=SwiftFilter.uw1)
        for cr in uw1_count_rates_vs_r.count_rates
    ]
    uvv_magnitudes = [
        magnitude_from_count_rate(count_rate=cr, filter_type=SwiftFilter.uvv)
        for cr in uvv_count_rates_vs_r.count_rates
    ]

    radii_count_rates_and_dust_color = product(
        zip(
            aperture_radii_pix,
            uw1_count_rates_vs_r.count_rates,
            uvv_count_rates_vs_r.count_rates,
        ),
        dust_rednesses,
    )

    print(
        f"Computing flux and water production at dust reddening {dust_rednesses[0]} through {dust_rednesses[-1]} ..."
    )
    flux_OH = {}
    num_OH = {}
    q_H2O = {}
    for (r, uw1_cr, uvv_cr), dust_redness in tqdm(
        radii_count_rates_and_dust_color,
        total=len(dust_rednesses) * len(aperture_radii_pix),
    ):
        flux_OH[r, dust_redness] = OH_flux_from_count_rate(
            uw1=uw1_cr,
            uvv=uvv_cr,
            beta=beta_parameters[dust_redness],
        )
        num_OH[r, dust_redness] = flux_OH_to_num_OH(
            flux_OH=flux_OH[r, dust_redness],
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta_au,
        )
        # TODO: backend hard-coded
        q_H2O[r, dust_redness] = num_OH_to_Q_vectorial(
            helio_r_au=helio_r_au, num_OH=num_OH[r, dust_redness], model_backend="rust"
        )

    radii_count_rates_mags_and_dust_color = product(
        zip(
            aperture_radii_pix,
            uw1_count_rates_vs_r.count_rates,
            uvv_count_rates_vs_r.count_rates,
            uw1_magnitudes,
            uvv_magnitudes,
        ),
        dust_rednesses,
    )

    print("Gathering results...")
    q_vs_aperture_results_list = [
        QvsApertureRadiusEntry(
            aperture_r_pix=r,
            aperture_r_km=r * km_per_pix,
            dust_redness=float(dust_redness),
            counts_uw1=uw1_cr.value,
            counts_uw1_err=uw1_cr.sigma,
            snr_uw1=uw1_cr.value / uw1_cr.sigma,
            counts_uvv=uvv_cr.value,
            counts_uvv_err=uvv_cr.sigma,
            snr_uvv=uvv_cr.value / uvv_cr.sigma,
            magnitude_uw1=uw1_mag.value,
            magnitude_uw1_err=uw1_mag.sigma,
            magnitude_uvv=uvv_mag.value,
            magnitude_uvv_err=uvv_mag.sigma,
            flux_OH=flux_OH[r, dust_redness].value,
            flux_OH_err=flux_OH[r, dust_redness].sigma,
            num_OH=num_OH[r, dust_redness].value,
            num_OH_err=num_OH[r, dust_redness].sigma,
            q_H2O=q_H2O[r, dust_redness].value,
            q_H2O_err=q_H2O[r, dust_redness].sigma,
        )
        for (
            r,
            uw1_cr,
            uvv_cr,
            uw1_mag,
            uvv_mag,
        ), dust_redness in tqdm(radii_count_rates_mags_and_dust_color)
    ]

    return q_vs_aperture_results_list


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


def get_production_plateaus(
    sorted_q_vs_r: list[QvsApertureRadiusEntry],
) -> dict[DustReddeningPercent, list[ProductionPlateau]]:
    """
    sorted_q_vs_r should have been previously sorted by dust_redness to make sure the entries are contiguous so groupby() catches all of them
    """

    by_redness = lambda x: x.dust_redness

    q_plateau_list_dict = {}
    for dust_redness, q_vs_aperture_radius_entry_at_redness in groupby(
        sorted_q_vs_r, key=by_redness
    ):
        qvarear = list(q_vs_aperture_radius_entry_at_redness)

        q_plateau_list = find_production_plateaus(q_vs_aperture_radius_list=qvarear)
        q_plateau_list_dict[dust_redness] = q_plateau_list

    return q_plateau_list_dict


def q_vs_aperture_radius_at_epoch(
    epoch_subpipeline_files: EpochSubpipelineFiles,
    dust_rednesses: list[DustReddeningPercent],
) -> None:

    epoch_id = epoch_subpipeline_files.parent_epoch.epoch_id

    epoch_subpipeline_files.stacked_epoch.read_product_if_not_loaded()
    stacked_epoch = epoch_subpipeline_files.stacked_epoch.data
    if stacked_epoch is None:
        # TODO: error message
        return

    print(
        f"Starting analysis of {epoch_id}: observation at {np.mean(stacked_epoch.HELIO)} AU"
    )

    # TODO: select which method with menu
    # print("Selecting stacking method: sum")
    stacking_method = StackingMethod.summation

    # load background-subtracted images
    epoch_subpipeline_files.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].read_product_if_not_loaded()
    epoch_subpipeline_files.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].read_product_if_not_loaded()

    uw1_img = epoch_subpipeline_files.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].data.data
    uvv_img = epoch_subpipeline_files.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].data.data

    if uw1_img is None or uvv_img is None:
        print("Error loading background-subtracted images!")
        return

    epoch_subpipeline_files.background_analyses[
        SwiftFilter.uw1, stacking_method
    ].read_product_if_not_loaded()
    epoch_subpipeline_files.background_analyses[
        SwiftFilter.uvv, stacking_method
    ].read_product_if_not_loaded()
    uw1_bg = dict_to_background_result(
        epoch_subpipeline_files.background_analyses[
            SwiftFilter.uw1, stacking_method
        ].data
    )
    uvv_bg = dict_to_background_result(
        epoch_subpipeline_files.background_analyses[
            SwiftFilter.uvv, stacking_method
        ].data
    )

    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        return

    # TODO: use the larger end of the plateau as the radius of an aperture on a completed vectorial model, and use *that* calculate total OH and match production

    q_vs_r = q_vs_aperture_radius(
        stacked_epoch=stacked_epoch,
        uw1_img=uw1_img,
        uvv_img=uvv_img,
        dust_rednesses=dust_rednesses,
        uw1_bg=uw1_bg,
        uvv_bg=uvv_bg,
        max_aperture_radius=150000 * u.km,  # type: ignore
        num_apertures=30,
    )

    by_dust_redness = lambda x: x.dust_redness
    sorted_q_vs_r = sorted(q_vs_r, key=by_dust_redness)  # type: ignore

    q_plateau_list_dict = get_production_plateaus(sorted_q_vs_r=sorted_q_vs_r)

    # for dust_redness in dust_rednesses:
    #     print(f"{dust_redness=}")
    #     if q_plateau_list_dict[dust_redness] is not None:
    #         print(f"{q_plateau_list_dict[dust_redness]}")
    #     else:
    #         print("None found")

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    for dust_redness, q_vs_aperture_radius_entry_at_redness in groupby(sorted_q_vs_r, key=by_dust_redness):  # type: ignore
        qvarear = list(q_vs_aperture_radius_entry_at_redness)

        show_q_vs_aperture_with_plateaus(
            q_vs_aperture_radius_list=qvarear,
            q_plateau_list=q_plateau_list_dict[dust_redness],
            km_per_pix=km_per_pix,
        )

    df = dataframe_from_q_vs_aperture_radius_entry_list(sorted_q_vs_r)
    df.attrs.update(dust_plateau_list_dict_serialize(q_plateau_list_dict))  # type: ignore

    rprint("[green]Writing q vs aperture radius results ...[/green]")
    epoch_subpipeline_files.qh2o_vs_aperture_radius_analyses[stacking_method].data = df
    epoch_subpipeline_files.qh2o_vs_aperture_radius_analyses[stacking_method].write()


# TODO: show stacked images with aperture radii shaded in given the plateaus it finds
def qH2O_vs_aperture_radius_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available!")
        return

    # dust_rednesses = [DustReddeningPercent(0)] + [
    #     DustReddeningPercent(x) for x in np.geomspace(1, 50, num=50, endpoint=True)
    # ]

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(start=0.0, stop=50.0, num=51, endpoint=True)
    ]

    for parent_epoch in data_ingestion_files.epochs:
        esf = pipeline_files.epoch_subpipeline_from_parent_epoch(
            parent_epoch=parent_epoch
        )
        if esf is None:
            print(f"No subpipeline found for epoch {parent_epoch.product_path.name}")
            continue
        q_vs_aperture_radius_at_epoch(
            epoch_subpipeline_files=esf, dust_rednesses=dust_rednesses
        )
        print("")
