from itertools import product, groupby
from typing import TypeAlias

import numpy as np
from tqdm import tqdm
from rich import print as rprint
import astropy.units as u

from swift_comet_pipeline.aperture.aperture_count_rate import aperture_total_count_rate
from swift_comet_pipeline.aperture.plateau_detect import find_production_plateaus
from swift_comet_pipeline.aperture.plateau_serialize import (
    dust_plateau_list_dict_serialize,
)
from swift_comet_pipeline.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
    yaml_dict_to_background_result,
)
from swift_comet_pipeline.types.countrate_vs_aperture_radius import (
    CountrateVsApertureRadius,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.plateau import (
    ProductionPlateau,
    dict_to_production_plateau,
)
from swift_comet_pipeline.types.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
    dataframe_from_q_vs_aperture_radius_entry_list,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage
from swift_comet_pipeline.water_production.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.water_production.flux_OH import OH_flux_from_count_rate
from swift_comet_pipeline.water_production.num_OH_to_Q import num_OH_to_Q_vectorial


# TODO: make the visualization separate so we can load the csv and look at it later


ReddeningToProductionPlateauListDict: TypeAlias = dict[
    DustReddeningPercent, list[ProductionPlateau]
]


def count_rate_vs_aperture_radius(
    img: SwiftUVOTImage,
    bg: BackgroundResult,
    exposure_time_s: float,
    rs_pixels: list[float],
    use_tqdm: bool = False,
) -> CountrateVsApertureRadius:
    """
    Given an image, calculates the signal inside apertures of the given radii rs_pixels,
    and uses the background to propogate error.
    Optional text progress bar with use_tqdm.
    """

    comet_center = get_uvot_image_center(img=img)

    if use_tqdm is True:
        r_pix_list = tqdm(rs_pixels)
    else:
        r_pix_list = rs_pixels

    count_rates = [
        aperture_total_count_rate(
            img=img,
            aperture_center=comet_center,
            aperture_radius=r,
            background=bg,
            exposure_time_s=exposure_time_s,
        )
        for r in r_pix_list
    ]

    return CountrateVsApertureRadius(r_pixels=rs_pixels, count_rates=count_rates)


def q_vs_aperture_radius(
    epoch_summary: EpochSummary,
    uw1_img: SwiftUVOTImage,
    uvv_img: SwiftUVOTImage,
    dust_rednesses: list[DustReddeningPercent],
    uw1_bg: BackgroundResult,
    uvv_bg: BackgroundResult,
    max_aperture_radius: u.Quantity = 1.5e5 * u.km,  # type: ignore
    num_apertures: int = 300,
) -> list[QvsApertureRadiusEntry] | None:

    # TODO: document function

    # we are given a maximum aperture radius in physical units - translate to pixels
    r_pix_max = float(max_aperture_radius.to_value(u.km)) / epoch_summary.km_per_pix  # type: ignore
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
    uw1_count_rates_vs_r = count_rate_vs_aperture_radius(
        img=uw1_img,
        bg=uw1_bg,
        exposure_time_s=epoch_summary.uw1_exposure_time_s,
        rs_pixels=aperture_radii_pix,
        use_tqdm=True,
    )
    print("Calculating uvv counts ...")
    uvv_count_rates_vs_r = count_rate_vs_aperture_radius(
        img=uvv_img,
        bg=uvv_bg,
        exposure_time_s=epoch_summary.uvv_exposure_time_s,
        rs_pixels=aperture_radii_pix,
        use_tqdm=True,
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
            helio_r_au=epoch_summary.rh_au,
            helio_v_kms=epoch_summary.helio_v_kms,
            delta_au=epoch_summary.delta_au,
        )
        q_H2O[r, dust_redness] = num_OH_to_Q_vectorial(
            helio_r_au=epoch_summary.rh_au, num_OH=num_OH[r, dust_redness]
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
            aperture_r_km=r * epoch_summary.km_per_pix,
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


def get_production_plateaus(
    sorted_q_vs_r: list[QvsApertureRadiusEntry],
) -> ReddeningToProductionPlateauListDict:
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


def get_production_plateaus_from_yaml(
    yaml_dict: dict,
) -> ReddeningToProductionPlateauListDict:
    """
    Takes a ReddeningToProductionPlateauListDict that was stored as a dict in metadata and reconstructs into ReddeningToProductionPlateauListDict
    """

    q_plateau_list_dict: ReddeningToProductionPlateauListDict = {}

    for dust_redness, plateau_list_dict in yaml_dict.items():
        q_plateau_list_dict[dust_redness] = [
            dict_to_production_plateau(x) for x in plateau_list_dict
        ]

    return q_plateau_list_dict


def q_vs_aperture_radius_at_epoch(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    dust_rednesses: list[DustReddeningPercent],
) -> None:

    # TODO: document function

    if scp.exists(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
    ):
        print(f"Aperture analysis for {epoch_id} already exists - skipping.")
        # TODO: ask to re-calculate
        # wait_for_key()
        return

    # stacked_epoch = scp.get_product_data(
    #     pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    # )
    # assert stacked_epoch is not None

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert epoch_summary is not None

    print(f"Starting analysis of {epoch_id}: observation at {epoch_summary.rh_au} AU")

    # TODO: use pipeline utility functions
    uw1_img = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_img is not None
    uw1_img = uw1_img.data
    uvv_img = scp.get_product_data(
        pf=PipelineFilesEnum.background_subtracted_image,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_img is not None
    uvv_img = uvv_img.data

    if uw1_img is None or uvv_img is None:
        print("Error loading background-subtracted images!")
        return

    # TODO: use pipeline utility functions
    uw1_bg = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
        stacking_method=stacking_method,
    )
    assert uw1_bg is not None
    uw1_bg = yaml_dict_to_background_result(uw1_bg)
    uvv_bg = scp.get_product_data(
        pf=PipelineFilesEnum.background_determination,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_bg is not None
    uvv_bg = yaml_dict_to_background_result(uvv_bg)

    if uw1_bg is None or uvv_bg is None:
        print("Error loading background analysis!")
        return

    # TODO: use the larger end of the plateau as the radius of an aperture on a completed vectorial model, and use *that* calculate total OH and match production

    q_vs_r = q_vs_aperture_radius(
        # stacked_epoch=stacked_epoch,
        epoch_summary=epoch_summary,
        uw1_img=uw1_img,
        uvv_img=uvv_img,
        dust_rednesses=dust_rednesses,
        uw1_bg=uw1_bg,
        uvv_bg=uvv_bg,
        max_aperture_radius=150000 * u.km,  # type: ignore
        num_apertures=100,
    )

    by_dust_redness = lambda x: x.dust_redness
    sorted_q_vs_r = sorted(q_vs_r, key=by_dust_redness)  # type: ignore

    q_plateau_list_dict = get_production_plateaus(sorted_q_vs_r=sorted_q_vs_r)

    df = dataframe_from_q_vs_aperture_radius_entry_list(sorted_q_vs_r)
    df.attrs.update(dust_plateau_list_dict_serialize(q_plateau_list_dict))  # type: ignore

    rprint("[green]Writing q vs aperture radius results ...[/green]")
    ap_analysis_product = scp.get_product(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
    )
    assert ap_analysis_product is not None
    ap_analysis_product.data = df
    ap_analysis_product.write()
