import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit

from swift_comet_pipeline.modeling.vectorial_model import (
    num_OH_from_vectorial_model_result,
    water_vectorial_model,
)
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


def subtract_profiles(
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    # km_per_pix: float,
    dust_redness: DustReddeningPercent = 0.0,
) -> CometRadialProfile:
    # function assumes each radial profile is the same length radially
    assert len(uw1_profile.profile_axis_xs) == len(uvv_profile.profile_axis_xs)

    # should all be zero
    # print(uw1_profile.profile_axis_xs - uvv_profile.profile_axis_xs)

    beta = beta_parameter(dust_redness)

    uw1_params = get_filter_parameters(SwiftFilter.uw1)
    uvv_params = get_filter_parameters(SwiftFilter.uvv)

    # TODO: is this necessary? Ask Lucy
    uvv_to_uw1_cf = uvv_params["cf"] / uw1_params["cf"]

    subtracted_pixels = (
        uw1_profile.pixel_values - beta * uvv_to_uw1_cf * uvv_profile.pixel_values
    )
    # physical_rs = uw1_profile.profile_axis_xs * km_per_pix * u.km

    # pix = subtracted_pixels
    # rs = physical_rs

    # return rs, pix
    return CometRadialProfile(
        profile_axis_xs=uw1_profile.profile_axis_xs,
        pixel_values=subtracted_pixels,
        _xs=uw1_profile._xs,
        _ys=uw1_profile._ys,
        _radius=uw1_profile._radius,
        _theta=uw1_profile._theta,
        _comet_center=uw1_profile._comet_center,
    )


# TODO: the return type should be a dataclass
# def countrate_profile_to_surface_brightness(
#     countrate_profile: np.ndarray, km_per_pix: float, delta: u.Quantity
# ) -> np.ndarray:
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
    only_fit_beyond_r: u.Quantity = 5000 * u.km,  # type: ignore
):
    km_per_pix = np.mean(stacked_epoch.KM_PER_PIX)
    delta = np.mean(stacked_epoch.OBS_DIS) * u.AU  # type: ignore
    helio_v = np.mean(stacked_epoch.HELIO_V) * (u.km / u.s)  # type: ignore
    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

    subtracted_profile = subtract_profiles(
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
    )

    subtracted_profile_rs_km = subtracted_profile.profile_axis_xs * km_per_pix

    # limit fitting to r > only_fit_beyond_r
    profile_mask = subtracted_profile_rs_km > only_fit_beyond_r.to(u.km).value  # type: ignore

    masked_profile_rs_km = subtracted_profile_rs_km[profile_mask]
    masked_profile_pixels = subtracted_profile.pixel_values[profile_mask]

    surface_brightness_profile = countrate_profile_to_surface_brightness(
        countrate_profile=masked_profile_pixels, km_per_pix=km_per_pix, delta=delta
    )

    comet_column_density = surface_brightness_profile_to_column_density(
        surface_brightness_profile=surface_brightness_profile,
        delta=delta,
        helio_v=helio_v,
        helio_r=helio_r,
    )

    model_Q = 1e29 / u.s  # type: ignore

    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r, model_backend="rust")
    num_oh = num_OH_from_vectorial_model_result(vmr=vmr)
    print(f"Inside fitting: {num_oh=}")

    vectorial_model_column_density = vmr.column_density_interpolation(
        (masked_profile_rs_km * u.km).to(u.m).value  # type: ignore
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

    ccd = comet_column_density.to(1 / u.cm**2).value  # type: ignore
    vcd = (
        vectorial_model_column_density.to(1 / u.cm**2).value  # type: ignore
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

    return masked_profile_rs_km, ccd, vcd


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
    vcds = {}
    dust_rednesses = np.linspace(0.0, 100.0, num=11, endpoint=True)
    for dust_redness in dust_rednesses:
        rs[dust_redness], ccds[dust_redness], vcds[dust_redness] = find_q_at_redness(
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
        ax.plot(rs[dust_redness], vcds[dust_redness], label=f"vcd at {dust_redness}")
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_yscale("log")
    ax.set_ylabel("fragment column density, 1/cm^2")
    ax.legend()
    plt.show()

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore

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

    # TODO: pre perihelion, the column density might be linear and modeled by short-lived grains with low v_outflow
    # fit the profiles and find

    dust_color = dust_rednesses[2]
    linear_cd_fit(
        rs_km=rs[dust_color],
        cds_cm2=ccds[dust_color],
        within_r=50000 * u.km,
        helio_r_au=helio_r.to(u.AU).value,
    )


def linear_cd_fit(rs_km, cds_cm2, within_r: u.Quantity, helio_r_au) -> None:

    def lin(x: float, a: float, b: float) -> float:
        return a * x + b

    def haser_like(x: float, a: float, b: float):
        return a * x - b * np.log(x)

    r_mask = rs_km < (within_r.to(u.km).value)
    rs_fit_km = rs_km[r_mask]

    # rs_fit = rs_fit_km
    rs_fit = np.log(rs_fit_km)
    cd_fit = np.log(cds_cm2[r_mask])

    # popt, pcov = curve_fit(lin, rs_fit, cd_fit, sigma=rs_fit**2)
    popt, pcov = curve_fit(lin, rs_fit, cd_fit)
    lifetime_s = -1 / (popt[0] * np.log(1.05))
    exp_life = np.exp(lifetime_s)
    print(
        f"{popt=}, {pcov=}, {lifetime_s=}, {exp_life=}, {lifetime_s / helio_r_au**2}, {exp_life/helio_r_au**2}"
    )

    popt_haser, pcov_haser = curve_fit(haser_like, rs_fit, cd_fit)
    lifetime_s_haser = -1 / (popt_haser[0] * np.log(1.05))
    exp_life_haser = np.exp(lifetime_s_haser)
    print(
        f"{popt_haser=}, {pcov_haser=}, {lifetime_s_haser=}, {exp_life_haser=}, {lifetime_s_haser / helio_r_au**2}, {exp_life_haser/helio_r_au**2}"
    )

    fit_cds = lin(rs_fit, a=popt[0], b=popt[1])
    rs_plot_km = np.exp(rs_fit)
    cd_plot_cm2 = np.exp(fit_cds)
    cd_extrap = np.exp(lin(np.log(rs_km), a=popt[0], b=popt[1]))

    fit_cds_haser = haser_like(rs_fit, a=popt_haser[0], b=popt_haser[1])
    cd_plot_haser = np.exp(fit_cds_haser)
    cd_extrap_haser = np.exp(
        haser_like(np.log(rs_km), a=popt_haser[0], b=popt_haser[1])
    )

    _, ax = plt.subplots()
    ax.plot(rs_km, cds_cm2, label="coldens")
    ax.plot(rs_plot_km, cd_plot_cm2, label="fit")
    ax.plot(rs_km, cd_extrap, label="extrapolated fit", alpha=0.5)
    ax.plot(rs_plot_km, cd_plot_haser, label="haser-like")
    ax.plot(rs_km, cd_extrap_haser, label="haser-like extrapolated")
    ax.set_xscale("log")
    ax.set_xlabel("distance r from nucleus, km")
    ax.set_yscale("log")
    ax.set_ylabel("log fragment column density, 1/cm^2")
    ax.legend()
    plt.show()
