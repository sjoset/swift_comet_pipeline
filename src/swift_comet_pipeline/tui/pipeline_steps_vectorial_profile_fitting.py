import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from matplotlib.colors import Normalize, LinearSegmentedColormap
import astropy.units as u
from astropy.time import Time
from astropy.visualization import ZScaleInterval

# from scipy.optimize import curve_fit

from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)
from swift_comet_pipeline.comet.column_density import (
    ColumnDensity,
)
from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import (
    VectorialModelFit,
    vectorial_fit,
)

# from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.observationlog.stacked_epoch import StackedEpoch
from swift_comet_pipeline.orbits.perihelion import find_perihelion

# from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.tui.tui_common import (
    get_selection,
    stacked_epoch_menu,
)
from swift_comet_pipeline.comet.comet_radial_profile import (
    radial_profile_from_dataframe_product,
)


# TODO: update this
# TODO: move this to extras?


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


# def column_density_ratio_plot(
#     stacked_epoch: Epoch,
#     dust_rednesses: list[DustReddeningPercent],
#     ccds: dict[DustReddeningPercent, ColumnDensity],
#     uw1_stack: SwiftUVOTImage,
#     uvv_stack: SwiftUVOTImage,
#     t_perihelion: Time,
# ) -> None:
#
#     # TODO: turn the image and label annotation into a function that takes positions as arguments
#
#     helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
#     delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
#
#     time_from_perihelion = Time(np.mean(stacked_epoch.MID_TIME)) - t_perihelion
#
#     zscale = ZScaleInterval()
#     uw1_vmin, uw1_vmax = zscale.get_limits(uw1_stack)
#     uvv_vmin, uvv_vmax = zscale.get_limits(uvv_stack)
#     uw1_norm = Normalize(vmin=uw1_vmin, vmax=uw1_vmax)
#     uvv_norm = Normalize(vmin=uvv_vmin, vmax=uvv_vmax)
#
#     # # study how the redness affects the column density as a function of radius
#     fig, ax = plt.subplots()
#     for dust_redness in dust_rednesses[1:]:
#         ax.plot(
#             ccds[dust_redness].rs_km,
#             ccds[dust_redness].cd_cm2 / ccds[dust_rednesses[0]].cd_cm2,
#             label=f"cd ratio at {dust_redness=}",
#         )
#     ax.set_xscale("log")
#     ax.set_xlabel("distance r from nucleus, km")
#     ax.set_ylabel("fragment column density ratio")
#     ax.legend()
#     fig.suptitle(
#         f"Rh: {helio_r.to_value(u.AU):1.4f} AU, Delta: {delta.to_value(u.AU):1.4f} AU,\nTime from perihelion: {time_from_perihelion.to_value(u.day)} days"  # type: ignore
#     )
#     uw1_offset_img, uw1_text_label = make_image_annotations(
#         img=uw1_stack, norm=uw1_norm, t="Filter: uw1"
#     )
#     uw1_offset_img.image.axes = ax
#     ab_uw1 = AnnotationBbox(
#         uw1_offset_img,
#         xy=(1, 1),
#         xycoords="axes fraction",
#         box_alignment=(0.5, 0.5),
#         bboxprops=dict(alpha=0.0),
#     )
#     tab_uw1 = AnnotationBbox(
#         uw1_text_label,
#         xy=(1.0, 0.91),
#         xycoords="axes fraction",
#         bboxprops=dict(alpha=0.0),
#     )
#     ax.add_artist(ab_uw1)
#     ax.add_artist(tab_uw1)
#     uvv_offset_img, uvv_text_label = make_image_annotations(
#         img=uvv_stack, norm=uvv_norm, t="Filter: uvv"
#     )
#     uvv_offset_img.image.axes = ax
#     ab_uvv = AnnotationBbox(
#         uvv_offset_img,
#         xy=(1, 1),
#         xycoords="axes fraction",
#         box_alignment=(0.5, 1.5),
#         bboxprops=dict(alpha=0.0),
#     )
#     tab_uvv = AnnotationBbox(
#         uvv_text_label,
#         # xy=(1.0, 0.87),
#         xy=(1.0, 0.65),
#         xycoords="axes fraction",
#         bboxprops=dict(alpha=0.0),
#     )
#     ax.add_artist(ab_uvv)
#     ax.add_artist(tab_uvv)
#     plt.show()
#     plt.close()


def vectorial_fitting_plots(
    stacked_epoch: StackedEpoch,
    dust_rednesses: list[DustReddeningPercent],
    ccds: dict[DustReddeningPercent, ColumnDensity],
    vectorial_fits: dict[DustReddeningPercent, VectorialModelFit],
    uw1_stack: SwiftUVOTImage,
    uvv_stack: SwiftUVOTImage,
    fit_begin_r: u.Quantity,
    fit_end_r: u.Quantity,
    t_perihelion: Time,
) -> None:

    dust_cmap = LinearSegmentedColormap.from_list(
        name="custom", colors=["#8e8e8e", "#bb0000"], N=(len(dust_rednesses) + 1)
    )
    dust_line_colors = dust_cmap(
        np.array(dust_rednesses).astype(np.float32) / DustReddeningPercent(100.0)
    )

    # TODO: lightly shade the fitted region like we do with detected plateaus

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    # TODO: get perihelion from orbital info
    time_from_perihelion = Time(np.mean(stacked_epoch.MID_TIME)) - t_perihelion

    zscale = ZScaleInterval()
    uw1_vmin, uw1_vmax = zscale.get_limits(uw1_stack)
    uvv_vmin, uvv_vmax = zscale.get_limits(uvv_stack)
    uw1_norm = Normalize(vmin=uw1_vmin, vmax=uw1_vmax)
    uvv_norm = Normalize(vmin=uvv_vmin, vmax=uvv_vmax)

    # how well does the near-nucleus fit vectorial model match the comet column density?
    fig, ax = plt.subplots()
    for dust_redness, line_color in zip(dust_rednesses, dust_line_colors):
        ax.plot(
            ccds[dust_redness].rs_km,
            ccds[dust_redness].cd_cm2,
            label=f"cd at {dust_redness=}",
            color=line_color,
            alpha=0.65,
        )
        ax.plot(
            vectorial_fits[dust_redness].vectorial_column_density.rs_km,
            vectorial_fits[dust_redness].vectorial_column_density.cd_cm2,
            label=f"vcd at {dust_redness}",
            color=line_color,
            alpha=0.8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_ylabel("log fragment column density, 1/cm^2")
    ax.legend()
    fig.suptitle(
        f"Rh: {helio_r.to_value(u.AU):1.4f} AU, Delta: {delta.to_value(u.AU):1.4f} AU,\nTime from perihelion: {time_from_perihelion.to_value(u.day)} days\nfitting data from {fit_begin_r.to(u.km):1.3e} to {fit_end_r.to(u.km):1.3e}"  # type: ignore
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


def vectorial_fitting_step(swift_project_config: SwiftProjectConfig) -> None:
    # TODO: check for existence of files before trying to load them
    # pipeline_files = PipelineFiles(swift_project_config.project_path)

    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]

    # data_ingestion_files = pipeline_files.data_ingestion_files
    # epoch_subpipelines = pipeline_files.epoch_subpipelines

    # if data_ingestion_files.epochs is None:
    #     print("No epochs found!")
    #     return

    t_perihelion_list = find_perihelion(scp=scp)
    if t_perihelion_list is None:
        print("Could not find time of perihelion!")
        return

    # if epoch_subpipelines is None:
    #     # TODO: better error message
    #     print("No epochs ready for this step!")
    #     return

    epoch_id_selected = stacked_epoch_menu(scp=scp)
    if epoch_id_selected is None:
        return

    # parent_epoch = stacked_epoch_menu(
    #     pipeline_files=pipeline_files, require_background_analysis_to_exist=True
    # )
    # if parent_epoch is None:
    #     return

    # epoch_subpipeline = pipeline_files.epoch_subpipeline_from_parent_epoch(
    #     parent_epoch=parent_epoch
    # )
    # if epoch_subpipeline is None:
    #     return

    # epoch_subpipeline.stacked_epoch.read()
    # stacked_epoch = epoch_subpipeline.stacked_epoch.data
    # if stacked_epoch is None:
    #     print("Error reading epoch!")
    #     return

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id_selected
    )
    assert stacked_epoch is not None

    # epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].read()
    # epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].read()
    # epoch_subpipeline.stacked_images[SwiftFilter.uw1, stacking_method].read()
    # epoch_subpipeline.stacked_images[SwiftFilter.uvv, stacking_method].read()

    # uw1_profile = radial_profile_from_dataframe_product(
    #     df=epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].data
    # )
    # uvv_profile = radial_profile_from_dataframe_product(
    #     df=epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].data
    # )

    uw1_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id_selected,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_profile is not None
    uw1_profile = radial_profile_from_dataframe_product(df=uw1_profile)
    uvv_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id_selected,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_profile is not None
    uvv_profile = radial_profile_from_dataframe_product(df=uvv_profile)

    # uw1_stack = epoch_subpipeline.stacked_images[
    #     SwiftFilter.uw1, stacking_method
    # ].data.data
    #
    # uvv_stack = epoch_subpipeline.stacked_images[
    #     SwiftFilter.uvv, stacking_method
    # ].data.data

    uw1_stack = scp.get_product_data(
        pf=PipelineFilesEnum.stacked_image,
        epoch_id=epoch_id_selected,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_stack is not None
    uw1_stack = uw1_stack.data
    uvv_stack = scp.get_product_data(
        pf=PipelineFilesEnum.stacked_image,
        epoch_id=epoch_id_selected,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_stack is not None
    uvv_stack = uvv_stack.data

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

    print("Running vectorial model...")
    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r)
    if vmr.column_density_interpolation is None:
        print(
            "No column density interpolation returned from vectorial model! This is a bug! Exiting."
        )
        exit(1)

    # TODO: make this an option in the config
    near_far_radius = 50000 * u.km  # type: ignore
    # dust_rednesses = np.linspace(0.0, 40.0, num=17, endpoint=True)
    # TODO: priority: should this be -100 to 100 for the bayesian step later?
    # TODO: no, this is a display-only step, so ask for the redness range here
    dust_rednesses = [
        DustReddeningPercent(x) for x in np.linspace(0.0, 100.0, num=11, endpoint=True)
    ]

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

    t_perihelion = t_perihelion_list[0].t_perihelion
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

    print(f"Heliocentric distance: {helio_r.to_value(u.AU):1.4f} AU")  # type: ignore
    print(f"Date: {Time(np.mean(stacked_epoch.MID_TIME))}")
    print(f"Perihelion: {t_perihelion}")
    print("Near-nucleus vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {near_fits[dust_redness].best_fit_Q:1.3e}\tErr: {near_fits[dust_redness].best_fit_Q_err:1.3e}"
        )

    print("Far from nucleus vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {far_fits[dust_redness].best_fit_Q:1.3e}\tErr: {far_fits[dust_redness].best_fit_Q_err:1.3e}"
        )

    print("Whole curve vectorial model fitting:")
    for dust_redness in dust_rednesses:
        print(
            f"Redness: {dust_redness}\tQ: {whole_fits[dust_redness].best_fit_Q:1.3e}\tErr: {whole_fits[dust_redness].best_fit_Q_err:1.3e}"
        )

    # column_density_ratio_plot(
    #     stacked_epoch=stacked_epoch,
    #     dust_rednesses=dust_rednesses,
    #     ccds=ccds,
    #     uw1_stack=uw1_stack,
    #     uvv_stack=uvv_stack,
    #     t_perihelion=t_perihelion,
    # )

    vectorial_fitting_plots(
        stacked_epoch=stacked_epoch,
        dust_rednesses=dust_rednesses,
        ccds=ccds,
        vectorial_fits=near_fits,
        uw1_stack=uw1_stack,
        uvv_stack=uvv_stack,
        fit_begin_r=0 * u.km,  # type: ignore
        fit_end_r=near_far_radius,
        t_perihelion=t_perihelion,
    )

    vectorial_fitting_plots(
        stacked_epoch=stacked_epoch,
        dust_rednesses=dust_rednesses,
        ccds=ccds,
        vectorial_fits=far_fits,
        uw1_stack=uw1_stack,
        uvv_stack=uvv_stack,
        fit_begin_r=near_far_radius,
        fit_end_r=np.max(ccds[DustReddeningPercent(0.0)].rs_km) * u.km,  # type: ignore
        t_perihelion=t_perihelion,
    )

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
    #
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
