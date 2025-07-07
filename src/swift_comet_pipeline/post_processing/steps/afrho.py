import pathlib
from functools import reduce

from astropy.table import QTable
import astropy.units as u
import numpy as np
import pandas as pd

from swift_comet_pipeline.dust.afrho import calculate_afrho
from swift_comet_pipeline.dust.halley_marcus import halley_marcus_curve_interpolation
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.post_processing.post_processing_steps import (
    AddEpochSummary,
    EpochPostProcessingStep,
    apply_epoch_post_processing_pipeline,
)
from swift_comet_pipeline.swift.magnitude_from_countrate import (
    magnitude_from_count_rate,
)
from swift_comet_pipeline.types.count_rate import CountRate
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter


class ApertureEpochPostProcessingStep(EpochPostProcessingStep):
    pass


class CalculateApertureAfrho(ApertureEpochPostProcessingStep):
    def __init__(
        self,
        aperture_afrho_cm_col="afrho_cm",
        aperture_afrho_cm_zero_col="afrho_cm_zero",
    ):
        super().__init__()
        # Names of our columns to add to the dataframe
        self.aperture_afrho_cm_col = aperture_afrho_cm_col
        self.aperture_afrho_cm_zero_col = aperture_afrho_cm_zero_col
        self.normalization_phase_deg = 0.0
        # self.output_columns = [
        #     self.aperture_afrho_cm_col,
        #     self.aperture_afrho_cm_zero_col,
        # ]

        self.required_input_columns = [
            "delta_au",
            "rh_au",
            "aperture_r_km",
            "magnitude_uvv",
            "phase_angle_deg",
        ]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        # TODO: we delete these columns later, so we should error out if they already exist so that
        # we don't delete existing work unexpectedly
        df = QTable.from_pandas(df_in.copy())
        df["delta"] = df_in.delta_au
        df["delta"].unit = u.AU  # type: ignore
        df["rh"] = df_in.rh_au
        df["rh"].unit = u.AU  # type: ignore
        df["rho"] = df_in.aperture_r_km
        df["rho"].unit = u.km  # type: ignore

        df[self.aperture_afrho_cm_col] = calculate_afrho(
            delta=df["delta"],  # type: ignore
            rh=df["rh"],  # type: ignore
            rho=df["rho"],  # type: ignore
            magnitude_uvv=df["magnitude_uvv"],  # type: ignore
        ).to_value(
            u.cm  # type: ignore
        )

        afrho_correction = halley_marcus_curve_interpolation(
            normalization_phase_deg=self.normalization_phase_deg
        )
        assert afrho_correction is not None
        df[self.aperture_afrho_cm_zero_col] = df[
            self.aperture_afrho_cm_col
        ] / afrho_correction(df_in.phase_angle_deg)

        df.remove_columns(["delta", "rh", "rho"])
        df_out = df.to_pandas()
        return df_out


def do_afrho_from_aperture_post_processing(
    scp: SwiftCometPipeline, stacking_method: StackingMethod, epoch_id: EpochID
) -> pd.DataFrame | None:

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    # get the aperture calculations
    ap_analysis_df = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
    )
    if ap_analysis_df is None:
        return None

    ap_pipeline_steps = [
        AddEpochSummary(epoch_summary=epoch_summary),
        CalculateApertureAfrho(),
    ]

    df = reduce(lambda df, pipe_step: pipe_step(df), ap_pipeline_steps, ap_analysis_df)

    return df


class ProfileEpochProcessingStep(EpochPostProcessingStep):
    pass


class AdjustColumnNames(ProfileEpochProcessingStep):
    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if "aperture_r_km" in df_in.columns:
            print(
                "Skipping AdjustColumnNames pipeline step: column aperture_r_km already present!"
            )
            return df_in

        # The radii that the profile is sampled from will become our aperture radii
        df = df_in.copy()
        df["aperture_r_km"] = df.r_km
        return df


class DoAperturePhotometry(ProfileEpochProcessingStep):
    def __init__(self):
        super().__init__()
        self.required_input_columns = ["r_pixel"]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()

        # calculate the total counts in a circular aperture using this radial profile
        df["inner_r_pix"] = df.r_pixel
        df["outer_r_pix"] = df.r_pixel.shift(-1)
        df["outer_r_pix"] = df.outer_r_pix.fillna(df.r_pixel)
        df["annulus_area_pix"] = np.pi * (df.outer_r_pix**2 - df.inner_r_pix**2)
        df["annulus_counts"] = df.annulus_area_pix * df.count_rate
        df["cumulative_counts"] = df.annulus_counts.cumsum()

        return df


class FilterUnusableRows(ProfileEpochProcessingStep):

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df = df[df["annulus_area_pix"] != 0.0]
        df = df[df["r_km"] != 0.0]
        return df  # type: ignore


class CalculateUVVMagnitude(ProfileEpochProcessingStep):
    # TODO: define necessary columns for this
    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        # calculate magnitude in UVV filter based on the counts in the apertures
        df["magnitude_uvv"] = df.cumulative_counts.apply(
            lambda x: magnitude_from_count_rate(
                CountRate(value=x, sigma=0.00001), filter_type=SwiftFilter.uvv
            )
        )
        # TODO: we don't keep track of the error here because of our dummy value for sigma of 0.00001 above
        df["magnitude_uvv"] = df.magnitude_uvv.apply(lambda x: x.value)

        return df


class CalculateProfileAfrho(ProfileEpochProcessingStep):
    def __init__(
        self,
        profile_afrho_cm_col="afrho_cm",
        profile_afrho_cm_zero_col="afrho_cm_zero",
    ):
        super().__init__()
        self.profile_afrho_cm_col = profile_afrho_cm_col
        self.profile_afrho_cm_zero_col = profile_afrho_cm_zero_col
        self.normalization_phase_deg = 0.0

        self.required_input_columns = [
            "delta_au",
            "rh_au",
            "aperture_r_km",
            "magnitude_uvv",
            "phase_angle_deg",
        ]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        # TODO: remove old code

        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = QTable.from_pandas(df_in.copy())

        df["delta"] = df["delta_au"]
        df["delta"].unit = u.AU  # type: ignore
        df["rh"] = df["rh_au"]
        df["rh"].unit = u.AU  # type: ignore
        df["rho"] = df["r_km"]
        df["rho"].unit = u.km  # type: ignore

        df[self.profile_afrho_cm_col] = calculate_afrho(
            delta=df["delta"],  # type: ignore
            rh=df["rh"],  # type: ignore
            rho=df["rho"],  # type: ignore
            magnitude_uvv=df["magnitude_uvv"],  # type: ignore
        ).to_value(
            u.cm  # type: ignore
        )

        afrho_correction = halley_marcus_curve_interpolation(
            normalization_phase_deg=self.normalization_phase_deg
        )
        assert afrho_correction is not None
        df[self.profile_afrho_cm_zero_col] = df[
            self.profile_afrho_cm_col
        ] / afrho_correction(df["phase_angle_deg"])

        return df.to_pandas()


def do_afrho_from_profile_post_processing(
    scp: SwiftCometPipeline, stacking_method: StackingMethod, epoch_id: EpochID
) -> pd.DataFrame | None:

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    # get the uvv profile
    uvv_profile = scp.get_product_data(
        pf=PipelineFilesEnum.extracted_profile,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
        stacking_method=stacking_method,
    )
    assert uvv_profile is not None
    uvv_profile = uvv_profile.sort_values("r_km").reset_index(drop=True)  # type: ignore

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    # TODO: enable post-process cache
    results_cache_path = (
        pathlib.Path(scp.pipeline_files.base_project_path)
        / "post_processing"
        / "afrho.csv"
    )

    ap_pipeline_steps = [
        AddEpochSummary(epoch_summary=epoch_summary),
        AdjustColumnNames(),
        DoAperturePhotometry(),
        FilterUnusableRows(),
        CalculateUVVMagnitude(),
        CalculateProfileAfrho(),
    ]

    df = apply_epoch_post_processing_pipeline(
        initial_dataframe=uvv_profile, ep=ap_pipeline_steps
    )

    return df
