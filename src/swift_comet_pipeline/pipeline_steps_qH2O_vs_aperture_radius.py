from itertools import product
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint
from tqdm import tqdm

from swift_comet_pipeline.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.flux_OH import OH_flux_from_count_rate, beta_parameter
from swift_comet_pipeline.num_OH_to_Q import num_OH_to_Q_vectorial
from swift_comet_pipeline.plateau_detect import plateau_detect
from swift_comet_pipeline.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.tui import stacked_epoch_menu, wait_for_key
from swift_comet_pipeline.comet_center import (
    compare_comet_center_methods,
)
from swift_comet_pipeline.comet_profile import comet_manual_aperture
from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.uvot_image import SwiftUVOTImage, get_uvot_image_center
from swift_comet_pipeline.count_rate import (
    CountRate,
    CountRatePerPixel,
    magnitude_from_count_rate,
)
from swift_comet_pipeline.epochs import Epoch
from swift_comet_pipeline.pipeline_files import PipelineFiles, PipelineProductType


def do_comet_photometry_at_img_center(
    img: SwiftUVOTImage, aperture_radius: float, bg: CountRatePerPixel
) -> CountRate:
    pix_center = get_uvot_image_center(img=img)
    ap_x, ap_y = pix_center.x, pix_center.y

    return comet_manual_aperture(
        img=img,
        aperture_x=ap_x,
        aperture_y=ap_y,
        aperture_radius=aperture_radius,
        bg=bg,
    )


def q_vs_aperture_radius(
    epoch: Epoch,
    uw1: SwiftUVOTImage,
    uvv: SwiftUVOTImage,
    dust_rednesses: List[DustReddeningPercent],
    bguw1: CountRatePerPixel,
    bguvv: CountRatePerPixel,
) -> pd.DataFrame:
    # TODO: schema for this dataframe and write it out
    helio_r_au = np.mean(epoch.HELIO)
    helio_v_kms = np.mean(epoch.HELIO_V)
    delta = np.mean(epoch.OBS_DIS)

    # TODO: can vectorial model give a better estimate? might be able to borrow the code for grid size here
    radius_km_guess = 1e5
    r_pix = int(radius_km_guess / np.mean(epoch.KM_PER_PIX))
    print(f"Guessing radius of 1e5 km or {r_pix} pixels")

    aperture_radii, r_step = np.linspace(1, 2 * r_pix, num=5 * r_pix, retstep=True)
    redness_to_beta = {x: beta_parameter(x) for x in dust_rednesses}

    count_uw1 = []
    count_uvv = []
    rs = []
    red_list = []
    qs = []
    flux = []
    oh = []
    progress_bar = tqdm(
        product(aperture_radii, dust_rednesses),
        total=(len(aperture_radii) * len(dust_rednesses)),
    )
    for ap_r, redness in progress_bar:
        rs.append(ap_r)
        red_list.append(redness)

        cuw1 = do_comet_photometry_at_img_center(
            img=uw1, aperture_radius=ap_r, bg=bguw1
        )
        cuvv = do_comet_photometry_at_img_center(
            img=uvv, aperture_radius=ap_r, bg=bguvv
        )

        count_uw1.append(cuw1)
        count_uvv.append(cuvv)

        flux_OH = OH_flux_from_count_rate(
            uw1=cuw1,
            uvv=cuvv,
            beta=redness_to_beta[redness],
        )
        flux.append(flux_OH)

        num_oh = flux_OH_to_num_OH(
            flux_OH=flux_OH,
            helio_r_au=helio_r_au,
            helio_v_kms=helio_v_kms,
            delta_au=delta,
        )
        oh.append(num_oh)

        q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
        qs.append(q)

    mags_uw1 = [magnitude_from_count_rate(x, SwiftFilter.uw1) for x in count_uw1]
    mags_uvv = [magnitude_from_count_rate(x, SwiftFilter.uvv) for x in count_uvv]
    df = pd.DataFrame(
        {
            "aperture_radius": rs,
            "dust_redness": list(map(lambda x: int(x), red_list)),
            "counts_uw1": [x.value for x in count_uw1],
            "sigma_counts_uw1": [x.sigma for x in count_uw1],
            "counts_uvv": [x.value for x in count_uvv],
            "sigma_counts_uvv": [x.sigma for x in count_uvv],
            "mag_uw1": [m.value for m in mags_uw1],
            "sigma_mag_uw1": [m.sigma for m in mags_uw1],
            "mag_uvv": [m.value for m in mags_uvv],
            "sigma_mag_uvv": [m.sigma for m in mags_uvv],
            "flux_OH": [f.value for f in flux],
            "sigma_flux_OH": [f.sigma for f in flux],
            "num_OH": [x.value for x in oh],
            "sigma_num_OH": [x.sigma for x in oh],
            "Q_H2O": [q.value for q in qs],
            "sigma_Q_H2O": [q.sigma for q in qs],
        }
    )
    df["snr_uw1"] = df.counts_uw1 / df.sigma_counts_uw1
    df["snr_uvv"] = df.counts_uvv / df.sigma_counts_uvv

    # qmask = df["Q_H2O"] > 0
    # print(df[qmask])

    print("")
    print("Plateau search in uw1 counts:")
    uw1_plateau_list = plateau_detect(
        ys=df.counts_uw1.values,
        xstep=float(r_step),
        smoothing=3,
        threshold=1e-2,
        min_length=5,
    )
    if len(uw1_plateau_list) > 0:
        for p in uw1_plateau_list:
            i = p.begin_index
            j = p.end_index
            print(
                f"Plateau between r = {df.loc[i].aperture_radius} to r = {df.loc[j].aperture_radius}"
            )
            plateau_slice = df[i:j]
            positive_Qs = plateau_slice[plateau_slice.Q_H2O > 0]
            if len(positive_Qs > 0):
                print(f"\tAverage Q_H2O: {np.mean(positive_Qs.Q_H2O)}")
            else:
                print("\tNo positive Q_H2O values found in this plateau")
    else:
        print("No plateaus in uw1 counts detected")

    print("")
    print("Plateau search in uvv counts:")
    uvv_plateau_list = plateau_detect(
        ys=df.counts_uvv.values,
        xstep=float(r_step),
        smoothing=5,
        threshold=5e-3,
        min_length=5,
    )
    if len(uvv_plateau_list) > 0:
        for p in uvv_plateau_list:
            i = p.begin_index
            j = p.end_index
            print(
                f"Plateau between r = {df.loc[i].aperture_radius} to r = {df.loc[j].aperture_radius}"
            )
            plateau_slice = df[i:j]
            positive_Qs = plateau_slice[plateau_slice.Q_H2O > 0]
            if len(positive_Qs > 0):
                # print(positive_Qs)
                print(f"\tAverage Q_H2O: {np.mean(positive_Qs.Q_H2O)}")
            else:
                print("\tNo positive Q_H2O values found in this plateau")
    else:
        print("No plateaus in uvv counts detected")
    print("")

    df_reds = [y for _, y in df.groupby("dust_redness")]
    for redness_df in df_reds:
        _, axs = plt.subplots(nrows=1, ncols=3)
        axs[2].set_yscale("log")

        redness_df.plot.line(
            x="aperture_radius", y="counts_uw1", subplots=True, ax=axs[0]
        )
        axs[0].errorbar(
            redness_df.aperture_radius,
            redness_df.counts_uw1,
            redness_df.sigma_counts_uw1,
        )
        axs[0].set_title(
            f"uw1 counts vs aperture radius with dust redness {redness_df.dust_redness.iloc[0]}"
        )
        redness_df.plot.line(
            x="aperture_radius", y="counts_uvv", subplots=True, ax=axs[1]
        )
        axs[1].errorbar(
            redness_df.aperture_radius,
            redness_df.counts_uvv,
            redness_df.sigma_counts_uvv,
        )
        redness_df.plot.line(x="aperture_radius", y="Q_H2O", subplots=True, ax=axs[2])
        axs[2].errorbar(
            redness_df.aperture_radius, redness_df.Q_H2O, redness_df.sigma_Q_H2O
        )

        # TODO: rewrite this to show plateau per redness
        # for p in uw1_plateau_list:
        #     i = p.begin_index
        #     j = p.end_index
        #     axs[0].axvspan(redness_df.loc[i].aperture_radius, redness_df.loc[j].aperture_radius, color="blue", alpha=0.1)  # type: ignore
        #
        # for p in uvv_plateau_list:
        #     i = p.begin_index
        #     j = p.end_index
        #     axs[1].axvspan(redness_df.loc[i].aperture_radius, redness_df.loc[j].aperture_radius, color="orange", alpha=0.1)  # type: ignore

        plt.show()

    return df


def qH2O_vs_aperture_radius_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    # select the epoch we want to process
    epoch_id = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_be=True
    )
    if epoch_id is None:
        return

    # load the epoch database
    epoch = pipeline_files.read_pipeline_product(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id
    )
    epoch_path = pipeline_files.get_product_path(
        PipelineProductType.stacked_epoch, epoch_id=epoch_id
    )
    if epoch is None or epoch_path is None:
        print("Error loading epoch!")
        wait_for_key()
        return

    print(
        f"Starting analysis of {epoch_path.stem}: observation at {np.mean(epoch.HELIO)} AU"
    )

    # TODO: select which method with menu
    stacking_method = StackingMethod.summation

    # load background-subtracted images
    uw1 = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv = pipeline_files.read_pipeline_product(
        PipelineProductType.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uw1 is None or uvv is None:
        print("Error loading background-subtracted images!")
        wait_for_key()
        return

    # load the background analysis values and uncertainties for error propogation
    uw1_bg = pipeline_files.read_pipeline_product(
        PipelineProductType.background_analysis,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    uvv_bg = pipeline_files.read_pipeline_product(
        PipelineProductType.background_analysis,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        wait_for_key()
        return

    compare_comet_center_methods(uw1, uvv)

    q_vs_r = q_vs_aperture_radius(
        epoch=epoch,
        uw1=uw1,
        uvv=uvv,
        dust_rednesses=[DustReddeningPercent(x) for x in [0, 10, 20, 30, 40]],
        # dust_rednesses=[DustReddeningPercent(reddening=x) for x in [0, 10]],
        # dust_rednesses=[DustReddeningPercent(reddening=x) for x in [40]],
        bguw1=uw1_bg.count_rate_per_pixel,
        bguvv=uvv_bg.count_rate_per_pixel,
    )

    # fit_inverse_r(uw1)
    # fit_inverse_r(uvv)

    # peak_search_aperture_radius = 40
    # pix_center = get_uvot_image_center(img=uvv)

    # TODO: check if uw1 and uvv peak are the same, then use peak, otherwise, image center
    # search_aperture = CircularAperture(
    #     (pix_center.x, pix_center.y), r=peak_search_aperture_radius
    # )
    # peak = find_comet_center(
    #     img=uw1,
    #     method=CometCenterFindingMethod.aperture_peak,
    #     search_aperture=search_aperture,
    # )

    # imgs = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}

    # radius_estimation = {}
    #
    # for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
    #     peak = pix_center
    #     # test_circular_aperture_vs_donut_stack(img=imgs[filter_type])
    #
    #     count_list = []
    #     ecr_list = []
    #     thetas = np.linspace(0, np.pi, num=50, endpoint=False)
    #     for theta in thetas:
    #         comet_profile = count_rate_profile(
    #             img=imgs[filter_type],
    #             comet_center=peak,
    #             theta=theta,
    #             r=profile_radius,
    #         )
    #         ccr = count_rate_from_count_rate_profile(
    #             comet_profile, bgresults[filter_type].count_rate_per_pixel
    #         )
    #         # print(f"From profile at {theta=}: {ccr}")
    #         count_list.append(ccr)
    #
    #         ecr = estimate_comet_radius_at_angle(
    #             img=imgs[filter_type],
    #             comet_center=peak,
    #             radius_guess=profile_radius,
    #             theta=theta,
    #         )
    #         # print(f"Estimated comet radius: {ecr}")
    #         ecr_list.append(ecr)
    #
    #     print("Using 50 cuts of comet profile at different angles:")
    #     cl_mean = np.mean([x.value for x in count_list])
    #     cl_median = np.median([x.value for x in count_list])
    #     cl_std = np.std([x.value for x in count_list])
    #     print(
    #         f"Count rate summary: avg = {cl_mean}, median = {cl_median}, std = {cl_std}"
    #     )
    #     print(
    #         f"Estimated comet radius: avg = {np.mean(ecr_list)}, median = {np.median(ecr_list)}, std = {np.std(ecr_list)}"
    #     )
    #     print("")
    #
    #     # cfe_ap = CircularAperture((peak.x, peak.y), np.mean(ecr_list))
    #     # cfe_stats = ApertureStats(imgs[filter_type], cfe_ap)
    #     # print(f"Counts in aperture of average radius from estimation: {cfe_stats.sum}")
    #
    #     radius_estimation[filter_type] = np.mean(ecr_list)
    #
    #     theta_list, radii_list = estimate_comet_radius_by_angle(
    #         img=imgs[filter_type],
    #         comet_center=peak,
    #         radius_guess=30,
    #     )
    #
    #     rl_ap = CircularAperture((peak.x, peak.y), np.mean(radii_list))
    #     rl_stats = ApertureStats(imgs[filter_type], rl_ap)
    #     print(f"Counts in aperture of average radius list: {rl_stats.sum}")
    #
    #     comet_profile = count_rate_profile(
    #         img=imgs[filter_type],
    #         comet_center=peak,
    #         theta=0,
    #         r=profile_radius,
    #     )
    #     fitted_model = fit_comet_profile_gaussian(comet_profile=comet_profile)
    #     plot_fitted_profile(
    #         comet_profile=comet_profile,
    #         fitted_model=fitted_model,
    #         sigma_threshold=4.0,
    #         plot_title=f"Comet profile along theta = 0, with estimated aperture radius for filter {filter_to_file_string(filter_type)}",
    #     )
    #
    # max_radius = np.max(
    #     [radius_estimation[SwiftFilter.uw1], radius_estimation[SwiftFilter.uvv]]
    # )
    # print(f"Taking the radius to be {max_radius} for both filters")
    # helio_r_au = np.mean(epoch.HELIO)
    # helio_v_kms = np.mean(epoch.HELIO_V)
    # delta = np.mean(epoch.OBS_DIS)
    #

    # uw1cr = comet_manual_aperture(
    #     imgs[SwiftFilter.uw1],
    #     aperture_x=peak.x,
    #     aperture_y=peak.y,
    #     aperture_radius=max_radius,
    #     bg=bgresults[SwiftFilter.uw1].count_rate_per_pixel,
    # )
    # uvvcr = comet_manual_aperture(
    #     imgs[SwiftFilter.uvv],
    #     aperture_x=peak.x,
    #     aperture_y=peak.y,
    #     aperture_radius=max_radius,
    #     bg=bgresults[SwiftFilter.uvv].count_rate_per_pixel,
    # )
    #
    # correction_factors = np.linspace(1.0, 3.0, num=50, endpoint=True)
    # correction_factors = np.append(correction_factors, [10.0, 100.0])
    # for correction_factor in correction_factors:
    #     # corr_uw1cr = copy.deepcopy(uw1cr)
    #     # corr_uw1cr.value = corr_uw1cr.value * correction_factor
    #
    #     flux_OH = OH_flux_from_count_rate(
    #         uw1=ValueAndStandardDev(
    #             value=uw1cr.value * correction_factor, sigma=uw1cr.sigma
    #         ),
    #         uvv=uvvcr,
    #         beta=beta_parameter(DustReddeningPercent(reddening=10)),
    #     )
    #
    #     num_oh = flux_OH_to_num_OH(
    #         flux_OH=flux_OH,
    #         helio_r_au=helio_r_au,
    #         helio_v_kms=helio_v_kms,
    #         delta_au=delta,
    #     )
    #
    #     q = num_OH_to_Q_vectorial(helio_r_au=helio_r_au, num_OH=num_oh)
    #
    #     print(
    #         f"Correction factor {correction_factor:7.6f}\t\tQ from best guess: {q.value:7.6e}"
    #     )

    # xs_list = []
    # ys_list = []
    # zs_list = []
    #
    # profile_radius = 30
    # # angles = [3 * np.pi / 2, np.pi / 2, np.pi / 4]
    # angles = np.linspace(0, np.pi, 50)
    # angles = angles[:-1]
    # for theta in angles:
    #     xs, ys, prof = extract_profile(
    #         img=uvv, r=profile_radius, theta=theta, plot_profile=False
    #     )
    #     xs_list.extend(xs)
    #     ys_list.extend(ys)
    #     zs_list.extend(prof)
    #
    # pix_center = get_uvot_image_center(img=uvv)
    # search_aperture = CircularAperture((pix_center.x, pix_center.y), r=profile_radius)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    #
    # comet_center = find_comet_center(
    #     img=uvv,
    #     method=CometCenterFindingMethod.aperture_peak,
    #     search_aperture=search_aperture,
    # )
    #
    # mx, my = np.meshgrid(
    #     np.linspace(
    #         comet_center[0] - profile_radius / 2, comet_center[0] + profile_radius / 2
    #     ),
    #     np.linspace(
    #         comet_center[1] - profile_radius / 2, comet_center[1] + profile_radius / 2
    #     ),
    # )
    # mz = interp.griddata((xs_list, ys_list), zs_list, (mx, my), method="linear")
    #
    # ax1.plot_surface(mx, my, mz, cmap="magma")
    #
    # # ax.plot_trisurf(xs_list, ys_list, zs_list, cmap="magma")
    # ax2.contourf(mx, my, mz, cmap="magma")
    #
    # plt.show()

    ####
    # save q vs r
    ####
    # q_prod = pipeline_files.analysis_qh2o_products[epoch_path]
    # q_prod.data_product = q_vs_r
    # q_prod.save_product()

    rprint("[green]Writing q vs aperture radius results ...[/green]")
    pipeline_files.write_pipeline_product(
        PipelineProductType.qh2o_vs_aperture_radius,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
        data=q_vs_r,
    )
    rprint("[green]Done[/green]")
    wait_for_key()
    # next step in pipeline should be to decide redness and aperture radius?

    # decide radius --> inform vectorial model about extent of grid?

    # dust profile + column density fitting? --> redness
