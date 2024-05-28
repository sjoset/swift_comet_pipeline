import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from swift_comet_pipeline.observationlog.epoch import Epoch
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, get_filter_parameters
from swift_comet_pipeline.water_production.fluorescence_OH import (
    gfactor_1au,
)
from swift_comet_pipeline.water_production.flux_OH import (
    beta_parameter,
)
from swift_comet_pipeline.water_production.num_OH_to_Q import (
    num_OH_at_r_au_vectorial,
)
from swift_comet_pipeline.tui.tui_common import (
    get_selection,
    stacked_epoch_menu,
)
from swift_comet_pipeline.comet.comet_profile import (
    CometRadialProfile,
    radial_profile_from_dataframe_product,
)


def arcseconds_to_au(arcseconds: float, delta: u.Quantity):
    # TODO: document this and explain magic numbers
    return delta.to(u.AU).value * 2 * np.pi * arcseconds / (3600.0 * 360)  # type: ignore


# def show_subtracted_profile(
#     uw1_profile: CometRadialProfile,
#     uvv_profile: CometRadialProfile,
#     stacked_epoch: Epoch,
# ) -> None:
#
#     # redness: DustReddeningPercent = rpsp.dust_redness
#     redness = DustReddeningPercent(0.0)
#     beta = beta_parameter(redness)
#
#     uw1_params = get_filter_parameters(SwiftFilter.uw1)
#     uvv_params = get_filter_parameters(SwiftFilter.uvv)
#
#     uvv_to_uw1_cf = uvv_params["cf"] / uw1_params["cf"]
#
#     # convert from count rate to flux
#     subtracted_pixels = (
#         uw1_profile.pixel_values - beta * uvv_to_uw1_cf * uvv_profile.pixel_values
#     )
#     positive_mask = subtracted_pixels > 0
#     physical_rs = uw1_profile.profile_axis_xs * km_per_pix
#
#     pix = subtracted_pixels[positive_mask]
#     rs = physical_rs[positive_mask]
#
#     # print(f"{base_q_per_s=}, running with Q={3*base_q_per_s}")
#     model_Q = 1e29
#     _, coma = num_OH_at_r_au_vectorial(base_q=model_Q / u.s, helio_r=rh * u.AU)
#     vectorial_values = (base_q_per_s / model_Q) * coma.vmr.column_density_interpolation(
#         rs * 1000
#     )
#
#     # convert pixel signal to column density
#     pixel_area_cm2 = (km_per_pix / 1.0e5) ** 2
#     # TODO: the 1.0 arcseconds should reflect the mode the image was taken with
#     pixel_side_length_cm = (
#         arcseconds_to_au(arcseconds=1.0, delta=delta_au) * u.AU
#     ).to_value(u.cm)
#     pixel_area_cm2 = pixel_side_length_cm**2
#     # print(f"{pixel_area_cm2=}")
#
#     # surface brightness = count rate per unit area
#     surf_brightness = pix / pixel_area_cm2
#
#     alpha = 1.2750906353215913e-12
#     flux = surf_brightness * alpha
#     delta_in_cm = (delta_au * u.AU).to_value(u.cm)
#     lumi = flux * 4 * np.pi * delta_in_cm**2
#
#     gfactor = gfactor_1au(helio_v_kms=helio_v_kms) / rh**2
#     cdens = lumi / gfactor
#
#     # print(cdens / vectorial_values)
#     # print(f"{delta_au=}\t{delta_in_cm=}\t{gfactor=}\t{rh=}")
#
#     fig, ax = plt.subplots()
#     ax.plot(rs, vectorial_values / 10000, label="vect")
#     ax.plot(rs, cdens, label="img")
#     # ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.legend()
#     plt.show()


def subtract_profiles(
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    km_per_pix: float,
    dust_redness: DustReddeningPercent = 0.0,
):
    # function assumes each radial profile is the same length radially
    assert len(uw1_profile.profile_axis_xs) == len(uvv_profile.profile_axis_xs)

    beta = beta_parameter(dust_redness)

    uw1_params = get_filter_parameters(SwiftFilter.uw1)
    uvv_params = get_filter_parameters(SwiftFilter.uvv)

    # TODO: is this necessary? Ask Lucy
    uvv_to_uw1_cf = uvv_params["cf"] / uw1_params["cf"]

    subtracted_pixels = (
        uw1_profile.pixel_values - beta * uvv_to_uw1_cf * uvv_profile.pixel_values
    )
    physical_rs = uw1_profile.profile_axis_xs * km_per_pix * u.km  # type: ignore

    pix = subtracted_pixels
    rs = physical_rs

    # TODO: maybe we should do this?
    # positive_mask = subtracted_pixels > 0
    # pix = subtracted_pixels[positive_mask]
    # rs = physical_rs[positive_mask]

    return rs, pix


# TODO: the return type should be a dataclass
# TODO: is this really surface brightness at this point or should we call it something else?
def countrate_profile_to_surface_brightness(
    countrate_profile: np.ndarray, km_per_pix: float, delta: u.Quantity
) -> np.ndarray:

    # convert pixel signal to column density
    pixel_area_cm2 = (km_per_pix / 1.0e5) ** 2
    # TODO: the 1.0 arcseconds should reflect the mode the image was taken with
    pixel_side_length_cm = (
        arcseconds_to_au(arcseconds=1.0, delta=delta) * u.AU  # type: ignore
    ).to_value(
        u.cm  # type: ignore
    )
    pixel_area_cm2 = pixel_side_length_cm**2

    # surface brightness = count rate per unit area
    surface_brightness_profile = countrate_profile / pixel_area_cm2

    return surface_brightness_profile


# TODO: add decorators to enforce the arguments are the correct Quantity
# TODO: the return type should specify that
def surface_brightness_profile_to_column_density(
    surface_brightness_profile: np.ndarray,
    delta: u.Quantity,
    helio_v: u.Quantity,
    helio_r: u.Quantity,
) -> np.ndarray:

    delta_cm = delta.to(u.cm).value  # type: ignore
    helio_v_kms = helio_v.to(u.km / u.s).value  # type: ignore
    rh_au = helio_r.to(u.AU).value  # type: ignore

    # TODO: magic numbers
    # specific to OH
    alpha = 1.2750906353215913e-12
    flux = surface_brightness_profile * alpha
    lumi = flux * 4 * np.pi * delta_cm**2

    gfactor = gfactor_1au(helio_v_kms=helio_v_kms) / rh_au**2
    column_density = lumi / gfactor

    return column_density / (u.cm**2)  # type: ignore


def show_column_densities(
    rs, comet_column_density, vectorial_model_column_density
) -> None:

    mask = rs > 5000

    # fig, ax = plt.subplots()
    _, ax = plt.subplots()

    ax.plot(
        rs[mask], vectorial_model_column_density[mask], label="vectorial column density"
    )
    ax.plot(rs[mask], comet_column_density[mask], label="comet column density")
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_yscale("log")
    ax.set_ylabel("fragment column density, 1/cm^2")
    ax.legend()
    plt.show()


def find_q_at_redness(
    stacked_epoch: Epoch,
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    dust_redness: DustReddeningPercent,
):

    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    helio_v = np.mean(stacked_epoch.HELIO_V) * (u.km / u.s)  # type: ignore
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

    subtracted_profile_rs, subtracted_profile_pixels = subtract_profiles(
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        km_per_pix=km_per_pix,
        dust_redness=dust_redness,
    )

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        countrate_profile=subtracted_profile_pixels, km_per_pix=km_per_pix, delta=delta
    )

    comet_column_density = surface_brightness_profile_to_column_density(
        surface_brightness_profile=surface_brightness_profile,
        delta=delta,
        helio_v=helio_v,
        helio_r=helio_r,
    )

    model_Q = 1e29 / u.s  # type: ignore
    _, coma = num_OH_at_r_au_vectorial(base_q=model_Q, helio_r=helio_r)
    vectorial_model_column_density = coma.vmr.column_density_interpolation(
        subtracted_profile_rs.to(u.m).value  # type: ignore
    ) / (
        u.m**2  # type: ignore
    )

    ratio = (vectorial_model_column_density / comet_column_density).decompose()

    estimated_model_to_comet_q_ratio_with_median = np.median(ratio)
    estimated_comet_q_median = model_Q / estimated_model_to_comet_q_ratio_with_median

    print(f"\nDust redness {dust_redness}:")
    print(
        f"Estimated ratio of (model Q)/(comet Q) through median: {estimated_model_to_comet_q_ratio_with_median}"
    )
    print(f"Estimated comet production: {estimated_comet_q_median}")

    ccd = comet_column_density.to(1 / u.cm**2).value
    vcd = (
        vectorial_model_column_density.to(1 / u.cm**2).value
        / estimated_model_to_comet_q_ratio_with_median
    )

    percent_rmse = np.sqrt(np.mean(np.square((vcd - ccd) / vcd)))
    print(f"{percent_rmse=}")

    # show_column_densities(
    #     rs=subtracted_profile_rs.to(u.km).value,  # type: ignore
    #     comet_column_density=comet_column_density.to(1 / u.m**2).value,  # type: ignore
    #     vectorial_model_column_density=vectorial_model_column_density.to(
    #         1 / u.m**2  # type: ignore
    #     ).value
    #     / estimated_model_to_comet_q_ratio_with_median,
    # )

    return subtracted_profile_rs.to(u.km).value, ccd, vcd


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

    uw1_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uw1, stacking_method].data
    )
    uvv_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline.extracted_profiles[SwiftFilter.uvv, stacking_method].data
    )

    rs = {}
    ccds = {}
    dust_rednesses = np.linspace(0.0, 100.0, num=11, endpoint=True)
    for dust_redness in dust_rednesses:
        rs[dust_redness], ccds[dust_redness], _ = find_q_at_redness(
            stacked_epoch=stacked_epoch,
            uw1_profile=uw1_profile,
            uvv_profile=uvv_profile,
            dust_redness=dust_redness,
        )

    for dust_redness in dust_rednesses[1:]:
        cd_ratios = ccds[dust_redness] / ccds[dust_rednesses[0]]
        # print(f"{dust_redness=}: {cd_ratios}")
        print(f"avg: {np.mean(cd_ratios)}\tstd: {np.std(cd_ratios)}")

    # TODO: the redness seems to affect the subtracted column density near the nucleus the most - is that always true?
    _, ax = plt.subplots()
    for dust_redness in dust_rednesses[1:]:
        ax.plot(
            rs[dust_redness],
            ccds[dust_redness] / ccds[dust_rednesses[0]],
            label=f"cd ratio at {dust_redness=}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    # ax.set_yscale("log")
    ax.set_ylabel("fragment column density ratio")
    ax.legend()
    plt.show()
    plt.close()

    _, ax = plt.subplots()
    for dust_redness in dust_rednesses:
        ax.plot(
            rs[dust_redness],
            ccds[dust_redness],
            label=f"cd at {dust_redness=}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_yscale("log")
    ax.set_ylabel("fragment column density, 1/cm^2")
    ax.legend()
    plt.show()
