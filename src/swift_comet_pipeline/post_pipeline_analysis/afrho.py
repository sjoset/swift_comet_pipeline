import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import QTable

from swift_comet_pipeline.dust.afrho import calculate_afrho
from swift_comet_pipeline.dust.halley_marcus import *
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.epoch_summary import get_epoch_summary
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter


# TODO: we should have a better-defined return type for this
def calculate_afrho_from_apertures(
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    epoch_id: EpochID,
) -> pd.DataFrame | None:
    """
    Reads the results of the bayesian aperture analysis and adds the columns 'afrho_cm' and 'afrho_cm_zero',
    where afrho_cm_zero is Afrho (in cm), normalized to zero phase
    """

    afrho_correction = halley_marcus_curve_interpolation(normalization_phase_deg=0.0)
    if afrho_correction is None:
        print("Could not construct Halley-Marcus curve for Afrho phase normalization!")
        print("This is a bug!")
        return None

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    # get the aperture calculations
    ap_analysis_df_pandas = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
    )
    if ap_analysis_df_pandas is None:
        return None

    # construct a QTable from our dataframe because it support astropy units, and it's simpler to
    # calculate using those and convert when finished
    ap_analysis_df = QTable.from_pandas(ap_analysis_df_pandas)
    ap_analysis_df["delta"] = epoch_summary.delta_au
    ap_analysis_df["delta"].unit = u.AU  # type: ignore
    ap_analysis_df["rh"] = epoch_summary.rh_au
    ap_analysis_df["rh"].unit = u.AU  # type: ignore
    ap_analysis_df["rho"] = ap_analysis_df["aperture_r_km"]
    ap_analysis_df["rho"].unit = u.km  # type: ignore

    ap_analysis_df["afrho_cm"] = calculate_afrho(
        delta=ap_analysis_df["delta"],  # type: ignore
        rh=ap_analysis_df["rh"],  # type: ignore
        rho=ap_analysis_df["rho"],  # type: ignore
        magnitude_uvv=ap_analysis_df["magnitude_uvv"],  # type: ignore
    ).to_value(
        u.cm  # type: ignore
    )

    ap_analysis_df["afrho_cm_zero"] = ap_analysis_df["afrho_cm"] * afrho_correction(
        epoch_summary.phase_angle_deg
    )

    ap_analysis_df.remove_columns(["delta", "rh", "rho"])
    return ap_analysis_df.to_pandas()


def calculate_afrho_from_uvv_profile(
    scp: SwiftCometPipeline, stacking_method: StackingMethod, epoch_id: EpochID
) -> pd.DataFrame | None:

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    afrho_correction = halley_marcus_curve_interpolation(normalization_phase_deg=0.0)
    if afrho_correction is None:
        print("Could not construct Halley-Marcus curve for Afrho phase normalization!")
        print("This is a bug!")
        return None

    # get the uvv profile
    uvv_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    uvv_profile = uvv_profile.sort_values("r_km").reset_index(drop=True)  # type: ignore
    uvv_profile["aperture_r_km"] = uvv_profile["r_km"]

    # calculate the total counts in a circular aperture using this radial profile
    uvv_profile["inner_r_pix"] = uvv_profile["r_pixel"]
    uvv_profile["outer_r_pix"] = uvv_profile["r_pixel"].shift(-1)
    uvv_profile["outer_r_pix"] = uvv_profile["outer_r_pix"].fillna(
        uvv_profile["r_pixel"]
    )
    uvv_profile["annulus_area_pix"] = np.pi * (
        uvv_profile["outer_r_pix"] ** 2 - uvv_profile["inner_r_pix"] ** 2
    )
    uvv_profile["annulus_counts"] = (
        uvv_profile["annulus_area_pix"] * uvv_profile["count_rate"]
    )
    uvv_profile["cumulative_counts"] = uvv_profile["annulus_counts"].cumsum()

    # filter unusable rows
    uvv_profile = uvv_profile[uvv_profile["annulus_area_pix"] != 0.0]
    uvv_profile = uvv_profile[uvv_profile["r_km"] != 0.0]

    # calculate magnitude in UVV filter based on the counts in the apertures
    uvv_profile["magnitude_uvv"] = uvv_profile["cumulative_counts"].apply(
        lambda x: magnitude_from_count_rate(
            CountRate(value=x, sigma=0.00001), filter_type=SwiftFilter.uvv
        )
    )
    # TODO: we don't keep track of the error here because of our dummy value for sigma of 0.00001 above
    uvv_profile["magnitude_uvv"] = uvv_profile["magnitude_uvv"].apply(lambda x: x.value)

    afrho_df = QTable.from_pandas(uvv_profile)

    afrho_df["delta"] = epoch_summary.delta_au
    afrho_df["delta"].unit = u.AU  # type: ignore
    afrho_df["rh"] = epoch_summary.rh_au
    afrho_df["rh"].unit = u.AU  # type: ignore
    afrho_df["rho"] = afrho_df["r_km"]
    afrho_df["rho"].unit = u.km  # type: ignore

    afrho_df["afrho_cm"] = calculate_afrho(
        delta=afrho_df["delta"],  # type: ignore
        rh=afrho_df["rh"],  # type: ignore
        rho=afrho_df["rho"],  # type: ignore
        magnitude_uvv=afrho_df["magnitude_uvv"],  # type: ignore
    ).to_value(
        u.cm  # type: ignore
    )

    afrho_df["afrho_cm_zero"] = afrho_df["afrho_cm"] * afrho_correction(
        epoch_summary.phase_angle_deg
    )

    return afrho_df.to_pandas()
