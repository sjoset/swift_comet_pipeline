import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.colors import Normalize
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit

from swift_comet_pipeline.comet.column_density import (
    ColumnDensity,
    surface_brightness_profile_to_column_density,
)
from swift_comet_pipeline.comet.comet_count_rate_profile import CometCountRateProfile
from swift_comet_pipeline.comet.comet_surface_brightness_profile import (
    countrate_profile_to_surface_brightness,
)
from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import (
    datamode_to_pixel_resolution,
)
from swift_comet_pipeline.tui.tui_common import (
    get_selection,
    stacked_epoch_menu,
)
from swift_comet_pipeline.comet.comet_radial_profile import (
    CometRadialProfile,
    radial_profile_from_dataframe_product,
    subtract_profiles,
)


def calculate_comet_column_density(
    stacked_epoch: Epoch,
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    dust_redness: DustReddeningPercent,
    r_min: u.Quantity,
) -> ColumnDensity:
    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    helio_v = np.mean(stacked_epoch.HELIO_V) * (u.km / u.s)  # type: ignore
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    pixel_resolution = datamode_to_pixel_resolution(stacked_epoch.DATAMODE[0])

    subtracted_profile = subtract_profiles(
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
    )

    subtracted_profile_rs_km = subtracted_profile.profile_axis_xs * km_per_pix

    # limit fitting to r > only_fit_beyond_r
    profile_mask = subtracted_profile_rs_km > r_min.to(u.km).value  # type: ignore

    profile_rs_km = subtracted_profile_rs_km[profile_mask]
    countrate_profile: CometCountRateProfile = subtracted_profile.pixel_values[
        profile_mask
    ]

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        countrate_profile=countrate_profile,
        pixel_resolution=pixel_resolution,
        delta=delta,
    )

    comet_column_density_values = surface_brightness_profile_to_column_density(
        surface_brightness_profile=surface_brightness_profile,
        delta=delta,
        helio_v=helio_v,
        helio_r=helio_r,
    )

    comet_column_density = ColumnDensity(
        rs_km=profile_rs_km, cd_cm2=comet_column_density_values.to(1 / u.cm**2).value  # type: ignore
    )

    return comet_column_density


def make_image_annotations(img: np.ndarray, norm, t: str):
    offset_img = OffsetImage(
        img,
        origin="lower",
        norm=norm,
        cmap="magma",
        zoom=0.08,
        alpha=0.9,
    )
    text_area = TextArea(
        t,
        textprops=dict(alpha=0.8, color="white"),
    )

    return offset_img, text_area


def vectorial_fitting_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.project_path)

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    data_ingestion_files = pipeline_files.data_ingestion_files
    epoch_subpipelines = pipeline_files.epoch_subpipelines

    if data_ingestion_files.epochs is None:
        print("No epochs found!")
        return

    if epoch_subpipelines is None:
        # TODO: better error message
        print("No epochs ready for this step!")
        return

    parent_epoch = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    )
    if parent_epoch is None:
        return

    epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
        parent_epoch=parent_epoch
    )
    if epoch_subpipeline is None:
        return

    epoch_subpipeline.stacked_epoch.read()
    stacked_epoch = epoch_subpipeline.stacked_epoch.data
    if stacked_epoch is None:
        print("Error reading epoch!")
        return

    epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].read()
    epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].read()
    epoch_subpipeline.stacked_images[SwiftFilter.uw1, stacking_method].read()
    epoch_subpipeline.stacked_images[SwiftFilter.uvv, stacking_method].read()

    uw1_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].data
    )
    uvv_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].data
    )

    zscale = ZScaleInterval()
    uw1_stack = epoch_subpipeline.stacked_images[
        SwiftFilter.uw1, stacking_method
    ].data.data
    uw1_vmin, uw1_vmax = zscale.get_limits(uw1_stack)
    uw1_norm = Normalize(vmin=uw1_vmin, vmax=uw1_vmax)

    uvv_stack = epoch_subpipeline.stacked_images[
        SwiftFilter.uvv, stacking_method
    ].data.data
    uvv_vmin, uvv_vmax = zscale.get_limits(uvv_stack)
    uvv_norm = Normalize(vmin=uvv_vmin, vmax=uvv_vmax)

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    t_perihelion = Time("2015-11-15")
    time_from_perihelion = Time(np.mean(stacked_epoch.MID_TIME)) - t_perihelion

    print("Running vectorial model...")
    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r, model_backend="rust")
    if vmr.column_density_interpolation is None:
        print(
            "No column density interpolation returned from vectorial model! This is a bug! Exiting."
        )
        exit(1)

    near_far_radius = 50000 * u.km  # type: ignore
    # dust_rednesses = np.linspace(0.0, 20.0, num=5, endpoint=True)
    dust_rednesses = np.linspace(0.0, 100.0, num=11, endpoint=True)

    ccds = {}
    near_fits = {}
    far_fits = {}
    whole_fits = {}
    for dust_redness in dust_rednesses:
        ccds[dust_redness] = calculate_comet_column_density(
            stacked_epoch=stacked_epoch,
            uw1_profile=uw1_profile,
            uvv_profile=uvv_profile,
            dust_redness=dust_redness,
            r_min=1 * u.km,  # type: ignore
        )
        near_fits[dust_redness] = vectorial_fit(
            comet_column_density=ccds[dust_redness],
            model_Q=model_Q,
            vmr=vmr,
            r_fit_min=1 * u.km,  # type: ignore
            r_fit_max=near_far_radius,
        )
        far_fits[dust_redness] = vectorial_fit(
            comet_column_density=ccds[dust_redness],
            model_Q=model_Q,
            vmr=vmr,
            r_fit_min=near_far_radius,
            r_fit_max=1.0e10 * u.km,  # type: ignore
        )
        whole_fits[dust_redness] = vectorial_fit(
            comet_column_density=ccds[dust_redness],
            model_Q=model_Q,
            vmr=vmr,
            r_fit_min=1 * u.km,  # type: ignore
            r_fit_max=1.0e10 * u.km,  # type: ignore
        )

    print("Near-nucleus vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {near_fits[dust_redness].best_fit_Q}\tErr: {near_fits[dust_redness].best_fit_Q_err}"
        )

    print("Far from nucleus vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {far_fits[dust_redness].best_fit_Q}\tErr: {far_fits[dust_redness].best_fit_Q_err}"
        )

    print("Whole curve vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {whole_fits[dust_redness].best_fit_Q}\tErr: {whole_fits[dust_redness].best_fit_Q_err}"
        )

    # # study how the redness affects the column density as a function of radius
    fig, ax = plt.subplots()
    for dust_redness in dust_rednesses[1:]:
        ax.plot(
            ccds[dust_redness].rs_km,
            ccds[dust_redness].cd_cm2 / ccds[dust_rednesses[0]].cd_cm2,
            label=f"cd ratio at {dust_redness=}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_ylabel("fragment column density ratio")
    ax.legend()
    fig.suptitle(
        f"Delta: {delta.to_value(u.AU):1.2f} AU, Time from perihelion: {time_from_perihelion.to_value(u.day)} days"
    )
    uw1_offset_img, uw1_text_label = make_image_annotations(
        img=uw1_stack, norm=uw1_norm, t="Filter: uw1"
    )
    uw1_offset_img.image.axes = ax
    ab_uw1 = AnnotationBbox(
        uw1_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 0.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uw1 = AnnotationBbox(
        uw1_text_label,
        xy=(1.0, 0.91),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uw1)
    ax.add_artist(tab_uw1)
    uvv_offset_img, uvv_text_label = make_image_annotations(
        img=uvv_stack, norm=uvv_norm, t="Filter: uvv"
    )
    uvv_offset_img.image.axes = ax
    ab_uvv = AnnotationBbox(
        uvv_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 1.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uvv = AnnotationBbox(
        uvv_text_label,
        # xy=(1.0, 0.87),
        xy=(1.0, 0.65),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uvv)
    ax.add_artist(tab_uvv)
    plt.show()
    plt.close()

    # how well does the near-nucleus fit vectorial model match the comet column density?
    fig, ax = plt.subplots()
    for dust_redness in dust_rednesses:
        ax.plot(
            ccds[dust_redness].rs_km,
            ccds[dust_redness].cd_cm2,
            label=f"cd at {dust_redness=}",
        )
        ax.plot(
            near_fits[dust_redness].column_density.rs_km,
            near_fits[dust_redness].column_density.cd_cm2,
            label=f"vcd at {dust_redness}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_ylabel("log fragment column density, 1/cm^2")
    ax.legend()
    fig.suptitle(
        f"Delta: {delta.to_value(u.AU):1.2f} AU, Time from perihelion: {time_from_perihelion.to_value(u.day)} days\nOnly fitting data within {near_far_radius.to_value(u.km)} km"
    )
    uw1_offset_img, uw1_text_label = make_image_annotations(
        img=uw1_stack, norm=uw1_norm, t="Filter: uw1"
    )
    uw1_offset_img.image.axes = ax
    ab_uw1 = AnnotationBbox(
        uw1_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 0.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uw1 = AnnotationBbox(
        uw1_text_label,
        xy=(1.0, 0.91),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uw1)
    ax.add_artist(tab_uw1)
    uvv_offset_img, uvv_text_label = make_image_annotations(
        img=uvv_stack, norm=uvv_norm, t="Filter: uvv"
    )
    uvv_offset_img.image.axes = ax
    ab_uvv = AnnotationBbox(
        uvv_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 1.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uvv = AnnotationBbox(
        uvv_text_label,
        xy=(1.0, 0.65),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uvv)
    ax.add_artist(tab_uvv)
    plt.show()
    plt.close()

    # how well does the far-nucleus fit vectorial model match the comet column density?
    fig, ax = plt.subplots()
    for dust_redness in dust_rednesses:
        ax.plot(
            ccds[dust_redness].rs_km,
            ccds[dust_redness].cd_cm2,
            label=f"cd at {dust_redness=}",
        )
        ax.plot(
            far_fits[dust_redness].column_density.rs_km,
            far_fits[dust_redness].column_density.cd_cm2,
            label=f"vcd at {dust_redness}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_ylabel("log fragment column density, 1/cm^2")
    ax.legend()
    fig.suptitle(
        f"Delta: {delta.to_value(u.AU):1.2f} AU, Time from perihelion: {time_from_perihelion.to_value(u.day)} days\nOnly fitting data beyond {near_far_radius.to_value(u.km)} km"
    )
    uw1_offset_img, uw1_text_label = make_image_annotations(
        img=uw1_stack, norm=uw1_norm, t="Filter: uw1"
    )
    uw1_offset_img.image.axes = ax
    ab_uw1 = AnnotationBbox(
        uw1_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 0.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uw1 = AnnotationBbox(
        uw1_text_label,
        xy=(1.0, 0.91),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uw1)
    ax.add_artist(tab_uw1)
    uvv_offset_img, uvv_text_label = make_image_annotations(
        img=uvv_stack, norm=uvv_norm, t="Filter: uvv"
    )
    uvv_offset_img.image.axes = ax
    ab_uvv = AnnotationBbox(
        uvv_offset_img,
        xy=(1, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 1.5),
        bboxprops=dict(alpha=0.0),
    )
    tab_uvv = AnnotationBbox(
        uvv_text_label,
        xy=(1.0, 0.65),
        xycoords="axes fraction",
        bboxprops=dict(alpha=0.0),
    )
    ax.add_artist(ab_uvv)
    ax.add_artist(tab_uvv)
    plt.show()
    plt.close()

    # model_Q = 1e29 / u.s  # type: ignore
    # _, coma = num_OH_at_r_au_vectorial(
    #     base_q=model_Q, helio_r=helio_r, water_grains=False
    # )
    # vectorial_model_column_density = coma.vmr.column_density
    #
    # _, grain_coma = num_OH_at_r_au_vectorial(
    #     base_q=model_Q, helio_r=helio_r, water_grains=True
    # )
    # grain_column_density = grain_coma.vmr.column_density
    #
    # vcd = vectorial_model_column_density.to(1 / u.cm**2).value
    # gcd = grain_column_density.to(1 / u.cm**2).value
    # rs = coma.vmr.column_density_grid

    # _, ax = plt.subplots()
    # ax.plot(rs, vcd, label="vectorial")
    # ax.plot(rs, gcd, label="grain")
    # ax.plot(rs, vcd + gcd, label="grain+vectorial")
    # ax.set_xscale("log")
    # ax.set_xlabel("distance r from nucleus, km")
    # ax.set_yscale("log")
    # ax.set_ylabel("fragment column density, 1/cm^2")
    # ax.legend()
    # plt.show()

    # dust_color = dust_rednesses[8]
    # linear_cd_fit(
    #     column_density=ccds[dust_color],
    #     within_r=near_far_radius,
    #     helio_r_au=helio_r.to(u.AU).value,  # type: ignore
    #     dust_redness=dust_color,
    # )
    # dust_color = dust_rednesses[0]
    # linear_cd_fit(
    #     column_density=ccds[dust_color],
    #     within_r=near_far_radius,
    #     helio_r_au=helio_r.to(u.AU).value,  # type: ignore
    #     dust_redness=dust_color,
    # )


# def linear_cd_fit(
#     column_density: ColumnDensity,
#     within_r: u.Quantity,
#     helio_r_au,
#     dust_redness: DustReddeningPercent,
# ) -> None:
#     # pre perihelion, the column density might be linear in log-log space and modeled by short-lived grains with low v_outflow
#     # fit the profiles and find out how well this works
#
#     positive_mask = column_density.cd_cm2 > 0.0
#     rs_km = column_density.rs_km[positive_mask]
#     cds_cm2 = column_density.cd_cm2[positive_mask]
#
#     def lin(x: float, a: float, b: float) -> float:
#         return a * x + b
#
#     def haser_like(x: float, a: float, b: float):
#         return a * x - b * np.log(x)
#
#     r_mask = rs_km < (within_r.to(u.km).value)  # type: ignore
#     rs_fit_km = rs_km[r_mask]
#
#     # rs_fit = rs_fit_km
#     rs_fit = np.log(rs_fit_km)
#     cd_fit = np.log(cds_cm2[r_mask])
#
#     # popt, pcov = curve_fit(lin, rs_fit, cd_fit, sigma=rs_fit**2)
#     popt, pcov = curve_fit(lin, rs_fit, cd_fit)
#     # TODO: figure out how to calculate the lifetime properly
#     lifetime_s = -1 / (popt[0] * 1.05)
#     exp_life = np.exp(lifetime_s)
#     print(
#         f"{popt=}, {pcov=}, {lifetime_s=}, {exp_life=}, lifetime at 1 au: {lifetime_s / helio_r_au**2}, exp lifetime at 1 au: {exp_life/helio_r_au**2}"
#     )
#
#     popt_haser, pcov_haser = curve_fit(haser_like, rs_fit, cd_fit)
#     lifetime_s_haser = -1 / (popt_haser[0] * 1.05)
#     exp_life_haser = np.exp(lifetime_s_haser)
#     print(
#         f"{popt_haser=}, {pcov_haser=}, {lifetime_s_haser=}, {exp_life_haser=}, lifetime at 1 au: {lifetime_s_haser / helio_r_au**2}, exp lifetime at 1 au: {exp_life_haser/helio_r_au**2}"
#     )
#
#     fit_cds = lin(rs_fit, a=popt[0], b=popt[1])
#     rs_plot_km = np.exp(rs_fit)
#     cd_plot_cm2 = np.exp(fit_cds)
#     cd_extrap = np.exp(lin(np.log(rs_km), a=popt[0], b=popt[1]))
#
#     fit_cds_haser = haser_like(rs_fit, a=popt_haser[0], b=popt_haser[1])
#     cd_plot_haser = np.exp(fit_cds_haser)
#     cd_extrap_haser = np.exp(
#         haser_like(np.log(rs_km), a=popt_haser[0], b=popt_haser[1])
#     )
#
#     _, ax = plt.subplots()
#     ax.plot(rs_km, cds_cm2, label="coldens")
#     ax.plot(rs_plot_km, cd_plot_cm2, label="fit")
#     ax.plot(rs_km, cd_extrap, label="extrapolated fit", alpha=0.5)
#     ax.plot(rs_plot_km, cd_plot_haser, label="haser-like")
#     ax.plot(rs_km, cd_extrap_haser, label="haser-like extrapolated")
#     ax.set_xscale("log")
#     ax.set_xlabel("distance r from nucleus, km")
#     ax.set_yscale("log")
#     ax.set_ylabel("log fragment column density, 1/cm^2")
#     ax.legend()
#     ax.set_title(
#         f"Linear fit to log-log plot of column density at dust redness {dust_redness}%"
#     )
#     plt.show()
