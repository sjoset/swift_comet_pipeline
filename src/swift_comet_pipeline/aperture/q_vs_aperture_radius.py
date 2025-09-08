from itertools import product, groupby, accumulate

import numpy as np
from tqdm import tqdm
from rich import print as rprint
import astropy.units as u

from swift_comet_pipeline.aperture.aperture_count_rate import aperture_analysis
from swift_comet_pipeline.aperture.concentric_annuli import (
    make_concentric_annular_apertures,
)
from swift_comet_pipeline.aperture.plateau_detect import find_production_plateaus
from swift_comet_pipeline.aperture.plateau_serialize import (
    dust_plateau_list_dict_serialize,
)
from swift_comet_pipeline.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.pipeline_utils.get_uw1_and_uvv import (
    get_uw1_and_uvv_background_results,
    get_uw1_and_uvv_background_subtracted_images,
)
from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.types.background_result import (
    BackgroundResult,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.plateau import dict_to_production_plateau
from swift_comet_pipeline.types.plateau_list import ReddeningToProductionPlateauListDict
from swift_comet_pipeline.types.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
    dataframe_from_q_vs_aperture_radius_entry_list,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage
from swift_comet_pipeline.water_production.fluorescence_OH import flux_OH_to_num_OH
from swift_comet_pipeline.water_production.flux_OH import OH_flux_from_count_rate
from swift_comet_pipeline.water_production.num_OH_to_Q import (
    num_OH_within_r_to_Q_vectorial,
)
from swift_comet_pipeline.types.uw1_uvv_pair import uw1uvv_getter


def aperture_count_rate_analysis(
    epoch_summary: EpochSummary,
    uw1_count_rate: CountRate,
    uvv_count_rate: CountRate,
    aperture_radius_pix: float,
    dust_redness: DustReddeningPercent,
) -> QvsApertureRadiusEntry:

    uw1_magnitude = magnitude_from_count_rate(
        count_rate=uw1_count_rate, filter_type=SwiftFilter.uw1
    )
    uvv_magnitude = magnitude_from_count_rate(
        count_rate=uvv_count_rate, filter_type=SwiftFilter.uvv
    )

    flux_oh = OH_flux_from_count_rate(
        uw1=uw1_count_rate,
        uvv=uvv_count_rate,
        beta=beta_parameter(dust_redness),
    )
    num_oh = flux_OH_to_num_OH(
        flux_OH=flux_oh,
        helio_r_au=epoch_summary.rh_au,
        helio_v_kms=epoch_summary.helio_v_kms,
        delta_au=epoch_summary.delta_au,
    )
    # q_h2o = num_OH_to_Q_vectorial(helio_r_au=epoch_summary.rh_au, num_OH=num_oh)
    q_h2o = num_OH_within_r_to_Q_vectorial(
        helio_r_au=epoch_summary.rh_au,
        num_OH=num_oh,
        within_r=aperture_radius_pix * epoch_summary.km_per_pix * u.km,  # type: ignore
    )

    return QvsApertureRadiusEntry(
        aperture_r_pix=aperture_radius_pix,
        aperture_r_km=aperture_radius_pix * epoch_summary.km_per_pix,
        dust_redness=float(dust_redness),
        counts_uw1=uw1_count_rate.value,
        counts_uw1_err=uw1_count_rate.sigma,
        counts_uvv=uvv_count_rate.value,
        counts_uvv_err=uvv_count_rate.sigma,
        magnitude_uw1=uw1_magnitude.value,
        magnitude_uw1_err=uw1_magnitude.sigma,
        magnitude_uvv=uvv_magnitude.value,
        magnitude_uvv_err=uvv_magnitude.sigma,
        flux_OH=flux_oh.value,
        flux_OH_err=flux_oh.sigma,
        num_OH=num_oh.value,
        num_OH_err=num_oh.sigma,
        q_H2O=q_h2o.value,
        q_H2O_err=q_h2o.sigma,
    )


def q_vs_aperture_radius(
    epoch_summary: EpochSummary,
    uw1_img: SwiftUVOTImage,
    uvv_img: SwiftUVOTImage,
    dust_rednesses: list[DustReddeningPercent],
    uw1_bg: BackgroundResult,
    uvv_bg: BackgroundResult,
    max_aperture_radius: u.Quantity,
    num_apertures: int = 300,
) -> list[QvsApertureRadiusEntry] | None:

    comet_center_uw1 = get_uvot_image_center(img=uw1_img)
    comet_center_uvv = get_uvot_image_center(img=uvv_img)

    r_pix_max = float(max_aperture_radius.to_value(u.km)) / epoch_summary.km_per_pix  # type: ignore
    if r_pix_max < 1.0:
        print(
            "Physical maximum aperture size given results in aperture size less than a pixel!"
        )
        return None

    uw1_apertures = make_concentric_annular_apertures(
        ap_center=comet_center_uw1,
        min_radius=0.0,
        max_radius=r_pix_max,
        num_slices=num_apertures,
    )
    uvv_apertures = make_concentric_annular_apertures(
        ap_center=comet_center_uvv,
        min_radius=0.0,
        max_radius=r_pix_max,
        num_slices=num_apertures,
    )

    aperture_radii = [float(uw1_apertures[0].r)] + [  # type: ignore
        float(x.r_out) for x in uw1_apertures[1:]  # type: ignore
    ]
    drs = np.array([uw1_apertures[0].r]) + np.diff(aperture_radii)  # type: ignore

    print("Starting UVW1 counts ...")
    uw1_annular_analyses = [
        aperture_analysis(
            img=uw1_img,
            ap=ap,
            background=uw1_bg,
            exposure_time_s=epoch_summary.uw1_exposure_time_s,
        )
        for ap in tqdm(uw1_apertures)
    ]
    print("Starting UVV counts ...")
    uvv_annular_median_count_rates = [
        aperture_analysis(
            img=uvv_img,
            ap=ap,
            background=uvv_bg,
            exposure_time_s=epoch_summary.uvv_exposure_time_s,
        )
        for ap in tqdm(uvv_apertures)
    ]

    # calculate the total count rates in the aperture, pretending all pixels were instead the median pixel value
    # innermost aperture is actually a circle
    uw1_annular_count_rates = [
        uw1_annular_analyses[0].median_count_rate * np.pi * drs[0] ** 2
    ]
    uvv_annular_count_rates = [
        uvv_annular_median_count_rates[0].median_count_rate * np.pi * drs[0] ** 2
    ]
    # and the rest are annuli
    uw1_annular_count_rates.extend(
        [2 * np.pi * x * dr for x, dr in zip(uw1_annular_analyses[1:], drs[1:])]
    )
    uvv_annular_count_rates.extend(
        [
            2 * np.pi * x * dr
            for x, dr in zip(uvv_annular_median_count_rates[1:], drs[1:])
        ]
    )

    uw1_count_rates = list(accumulate(uw1_annular_count_rates))
    uvv_count_rates = list(accumulate(uvv_annular_count_rates))

    num_data_points = len(dust_rednesses) * (len(aperture_radii) - 1)

    print("Starting Q(H2O) calculations ...")
    ap_analysis_list = [
        aperture_count_rate_analysis(
            epoch_summary=epoch_summary,
            uw1_count_rate=uw1_cr,
            uvv_count_rate=uvv_cr,
            aperture_radius_pix=ap_radius,
            dust_redness=dust_redness,
        )
        for dust_redness, (uw1_cr, uvv_cr, ap_radius) in tqdm(
            product(
                dust_rednesses, zip(uw1_count_rates, uvv_count_rates, aperture_radii)  # type: ignore
            ),
            total=num_data_points,
        )
    ]

    return ap_analysis_list


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

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert epoch_summary is not None

    print(
        f"Starting analysis of {epoch_id}: observation at {epoch_summary.rh_au} AU ... "
    )

    img_pair = get_uw1_and_uvv_background_subtracted_images(
        scp=scp, epoch_id=epoch_id, stacking_method=stacking_method
    )
    assert img_pair is not None
    uw1_img, uvv_img = uw1uvv_getter(img_pair)

    bg_pair = get_uw1_and_uvv_background_results(
        scp=scp, epoch_id=epoch_id, stacking_method=stacking_method
    )
    assert bg_pair is not None
    uw1_bg, uvv_bg = uw1uvv_getter(bg_pair)

    print("Images loaded ... ")

    # TODO: these magic numbers belong in a config: user or internal?
    # TODO: the upper limit should depend on the pixel scale: close comets only have ~20,000 km worth of data in image
    q_vs_r = q_vs_aperture_radius(
        epoch_summary=epoch_summary,
        uw1_img=uw1_img,
        uvv_img=uvv_img,
        dust_rednesses=dust_rednesses,
        uw1_bg=uw1_bg,
        uvv_bg=uvv_bg,
        max_aperture_radius=200000 * u.km,  # type: ignore
        # max_aperture_radius=1000000 * u.km,  # type: ignore
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
