from itertools import product
from typing import List
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from swift_comet_pipeline.background.background_result import dict_to_background_result
from swift_comet_pipeline.comet.comet_center_finding import compare_comet_center_methods

# from swift_comet_pipeline.comet.plateau_detect import find_production_plateau
from swift_comet_pipeline.comet.plateau_detect import find_production_plateaus
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.tui.tui_common import stacked_epoch_menu
from swift_comet_pipeline.comet.comet_aperture import comet_manual_aperture
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage, get_uvot_image_center
from swift_comet_pipeline.swift.count_rate import (
    CountRatePerPixel,
    magnitude_from_count_rate,
)
from swift_comet_pipeline.water_production.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.water_production.flux_OH import (
    OH_flux_from_count_rate,
    beta_parameter,
)
from swift_comet_pipeline.water_production.num_OH_to_Q import num_OH_to_Q_vectorial
from swift_comet_pipeline.water_production.q_vs_aperture_radius import (
    QvsApertureRadiusEntry,
    q_vs_aperture_radius_entry_list_from_dataframe,
)


# TODO: move this somewhere else
def q_vs_aperture_radius(
    stacked_epoch: Epoch,
    uw1: SwiftUVOTImage,
    uvv: SwiftUVOTImage,
    dust_rednesses: List[DustReddeningPercent],
    bguw1: CountRatePerPixel,
    bguvv: CountRatePerPixel,
) -> pd.DataFrame:
    # TODO: schema for this dataframe and write it out
    helio_r_au = np.mean(stacked_epoch.HELIO)
    helio_v_kms = np.mean(stacked_epoch.HELIO_V)
    delta_au = np.mean(stacked_epoch.OBS_DIS)

    radius_km_guess = 1e5
    r_pix = int(radius_km_guess / np.mean(stacked_epoch.KM_PER_PIX))
    print(f"Guessing radius of {radius_km_guess} km or {r_pix} pixels")

    # aperture_radii, r_step = np.linspace(
    #     1, r_pix, num=np.round(r_pix / 5).astype(np.int32), retstep=True
    # )
    aperture_radii = np.linspace(
        1, r_pix * 1.5, num=np.round(r_pix).astype(np.int32) * 5
    )
    beta_parameters = {x: beta_parameter(x) for x in dust_rednesses}
    comet_center = get_uvot_image_center(img=uw1)
    radii_and_dust_rednesses = list(product(aperture_radii, dust_rednesses))

    uw1_counts = {}
    uvv_counts = {}
    # find the counts and magnitudes in uw1 and uvv as a function of aperture radius - these don't depend on the dust redness
    for r in tqdm(aperture_radii, unit="radii"):
        uw1_counts[r] = comet_manual_aperture(
            img=uw1, aperture_center=comet_center, aperture_radius=r, bg=bguw1
        )
        uvv_counts[r] = comet_manual_aperture(
            img=uvv, aperture_center=comet_center, aperture_radius=r, bg=bguvv
        )
    uw1_magnitudes = {
        r: magnitude_from_count_rate(x, SwiftFilter.uw1) for r, x in uw1_counts.items()
    }
    uvv_magnitudes = {
        r: magnitude_from_count_rate(x, SwiftFilter.uvv) for r, x in uvv_counts.items()
    }

    flux_OH = {}
    num_OH = {}
    q_H2O = {}
    for r, dust_redness in radii_and_dust_rednesses:
        flux_OH[r, dust_redness] = OH_flux_from_count_rate(
            uw1=uw1_counts[r],
            uvv=uvv_counts[r],
            beta=beta_parameters[dust_redness],
        )
        num_OH[r, dust_redness] = flux_OH_to_num_OH(
            flux_OH=flux_OH[r, dust_redness],
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta_au,
        )
        q_H2O[r, dust_redness] = num_OH_to_Q_vectorial(
            helio_r_au=helio_r_au, num_OH=num_OH[r, dust_redness], model_backend="rust"
        )

    q_vs_aperture_results_list = [
        QvsApertureRadiusEntry(
            aperture_r_pix=r,
            dust_redness=float(dust_redness),
            counts_uw1=uw1_counts[r].value,
            counts_uw1_err=uw1_counts[r].sigma,
            snr_uw1=uw1_counts[r].value / uw1_counts[r].sigma,
            counts_uvv=uvv_counts[r].value,
            counts_uvv_err=uvv_counts[r].sigma,
            snr_uvv=uvv_counts[r].value / uvv_counts[r].sigma,
            magnitude_uw1=uw1_magnitudes[r].value,
            magnitude_uw1_err=uw1_magnitudes[r].sigma,
            magnitude_uvv=uvv_magnitudes[r].value,
            magnitude_uvv_err=uvv_magnitudes[r].sigma,
            flux_OH=flux_OH[r, dust_redness].value,
            flux_OH_err=flux_OH[r, dust_redness].sigma,
            num_OH=num_OH[r, dust_redness].value,
            num_OH_err=num_OH[r, dust_redness].sigma,
            q_H2O=q_H2O[r, dust_redness].value,
            q_H2O_err=q_H2O[r, dust_redness].sigma,
        )
        for r, dust_redness in radii_and_dust_rednesses
    ]

    q_vs_aperture_results_dict = [asdict(qvsar) for qvsar in q_vs_aperture_results_list]

    df = pd.DataFrame(data=q_vs_aperture_results_dict)

    return df


def qH2O_vs_aperture_radius_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)
    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipeline_files = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipeline_files is None:
        print("No epochs available to stack!")
        return

    # select the epoch we want to process
    parent_epoch = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    )
    if parent_epoch is None:
        return

    epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        parent_epoch=parent_epoch
    )
    if epoch_subpipeline is None:
        # TODO: error message
        return

    epoch_id = epoch_subpipeline.parent_epoch.epoch_id

    epoch_subpipeline.stacked_epoch.read()
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        # TODO: error message
        return

    print(
        f"Starting analysis of {epoch_id}: observation at {np.mean(stacked_epoch.HELIO)} AU"
    )

    # TODO: select which method with menu
    print("Selecting stacking method: sum")
    stacking_method = StackingMethod.summation

    # load background-subtracted images
    epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].read()
    epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].read()

    uw1_img = epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uw1, stacking_method
    ].data.data
    uvv_img = epoch_subpipeline.background_subtracted_images[
        SwiftFilter.uvv, stacking_method
    ].data.data

    if uw1_img is None or uvv_img is None:
        print("Error loading background-subtracted images!")
        return

    epoch_subpipeline.background_analyses[SwiftFilter.uw1, stacking_method].read()
    epoch_subpipeline.background_analyses[SwiftFilter.uvv, stacking_method].read()
    uw1_bg = dict_to_background_result(
        epoch_subpipeline.background_analyses[SwiftFilter.uw1, stacking_method].data
    )
    uvv_bg = dict_to_background_result(
        epoch_subpipeline.background_analyses[SwiftFilter.uvv, stacking_method].data
    )

    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        return

    # compare_comet_center_methods(uw1_img, uvv_img)

    q_vs_r = q_vs_aperture_radius(
        stacked_epoch=stacked_epoch,
        uw1=uw1_img,
        uvv=uvv_img,
        dust_rednesses=[
            DustReddeningPercent(x) for x in np.linspace(0, 30, num=2, endpoint=True)
        ],
        bguw1=uw1_bg.count_rate_per_pixel,
        bguvv=uvv_bg.count_rate_per_pixel,
    )

    q_plateau_list_dict = {}
    df = q_vs_r
    df_reds = df.groupby("dust_redness")
    for dust_redness, redness_df in df_reds:
        print(f"{dust_redness=}")

        q_plateau_list = find_production_plateaus(
            q_vs_aperture_list=q_vs_aperture_radius_entry_list_from_dataframe(
                df=redness_df
            )
        )
        q_plateau_list_dict[dust_redness] = q_plateau_list

        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle(f"dust redness {dust_redness}%/nm")
        axs[2].set_yscale("log")

        redness_df.plot.line(
            x="aperture_r_pix", y="counts_uw1", subplots=True, ax=axs[0]
        )
        axs[0].errorbar(
            redness_df.aperture_r_pix,
            redness_df.counts_uw1,
            redness_df.counts_uw1_err,
        )
        axs[0].set_title(f"uw1 counts vs aperture radius")

        redness_df.plot.line(
            x="aperture_r_pix", y="counts_uvv", subplots=True, ax=axs[1]
        )
        axs[1].errorbar(
            redness_df.aperture_r_pix,
            redness_df.counts_uvv,
            redness_df.counts_uvv_err,
        )
        axs[1].set_title(f"uvv counts vs aperture radius")

        redness_df.plot.line(
            x="aperture_r_pix", y="q_H2O", subplots=True, ax=axs[2], logy=True
        )
        axs[2].errorbar(
            redness_df.aperture_r_pix, redness_df.q_H2O, redness_df.q_H2O_err
        )
        axs[2].set_title(f"water production vs aperture radius")

        if q_plateau_list is not None:
            for p in q_plateau_list:
                print(f"{p=}")
                axs[0].axvspan(p.begin_r, p.end_r, color="blue", alpha=0.1)
                axs[1].axvspan(p.begin_r, p.end_r, color="blue", alpha=0.1)
                axs[2].axvspan(p.begin_r, p.end_r, color="blue", alpha=0.1)
                axs[2].text(
                    x=p.begin_r,
                    y=p.end_q / 100,
                    s=f"{p.begin_q:1.2e}",
                    color="#886",
                    alpha=0.8,
                )
                axs[2].text(
                    x=p.end_r,
                    y=p.end_q / 10,
                    s=f"{p.end_q:1.2e}",
                    color="#688",
                    alpha=0.8,
                )

        plt.show()

    # rprint("[green]Writing q vs aperture radius results ...[/green]")
    # epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].data = q_vs_r
    # epoch_subpipeline.qh2o_vs_aperture_radius_analyses[stacking_method].write()
    # rprint("[green]Done[/green]")
