import pathlib
from dataclasses import asdict

import pandas as pd
import astropy.units as u

from swift_comet_pipeline.modeling.vectorial_fitting_reliable import (
    column_density_has_enough_coverage,
    column_density_larger_than_psf_threshold,
)
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.post_processing.column_density_above_background import (
    column_density_above_background,
)
from swift_comet_pipeline.post_processing.post_processing_steps import (
    AddDustRednessesStep,
    AddStackingMethodStep,
    CreateDataframeFromEpochSummary,
    EpochPostProcessingStep,
    apply_epoch_post_processing_pipeline,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod


__all__ = ["do_vectorial_fitting_reliability_post_processing"]


class ColumnDensityAboveBackgroundAnalysisStep(EpochPostProcessingStep):
    def __init__(self, scp: SwiftCometPipeline):
        super().__init__()
        self.scp = scp

        self.required_input_columns = ["epoch_id", "dust_redness", "stacking_method"]
        self.cd_bg_result_dataclass_column = "cd_bg_result_dataclass"

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.cd_bg_result_dataclass_column] = df.apply(
            lambda row: column_density_above_background(
                scp=self.scp,
                epoch_id=row.epoch_id,
                dust_redness=row.dust_redness,
                stacking_method=row.stacking_method,
            ),
            axis=1,
        )

        # extract only these variables from the stored dataclass into their own columns
        bg_cd_columns_to_keep = [
            "last_usable_r",
            "last_usable_cd",
            "background_oh_cd",
            "num_usable_pixels_in_profile",
        ]
        df[
            [
                "last_usable_r",
                "last_usable_cd",
                "background_oh_cd",
                "num_usable_pixels_in_profile",
            ]
        ] = df[self.cd_bg_result_dataclass_column].apply(
            lambda row: pd.Series(
                {k: v for k, v in asdict(row).items() if k in bg_cd_columns_to_keep}
            )
        )
        return df


class CheckSufficientColumnDensityCoverage(EpochPostProcessingStep):
    def __init__(self, vectorial_fitting_requires_km: float):
        super().__init__()
        self.vectorial_fitting_requires_km = vectorial_fitting_requires_km
        self.vectorial_fitting_requires_column = (
            "vectorial_fitting_spatial_coverage_satisfied"
        )

        self.required_input_columns = ["cd_bg_result_dataclass"]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            print(
                "Not all required columns for CheckSufficientColumnDensityCoverage present!"
            )
            print(f"Present: {df_in.columns=}")
            print(f"Need: {self.required_input_columns=}")
            return df_in

        df = df_in.copy()
        df[self.vectorial_fitting_requires_column] = df.apply(
            lambda row: column_density_has_enough_coverage(
                cd_bg=row.cd_bg_result_dataclass
            ),
            axis=1,
        )
        return df


class CheckProfileExtendsBeyondPSF(EpochPostProcessingStep):
    def __init__(self, num_psfs_required: float):
        super().__init__()
        self.num_psfs_required = num_psfs_required
        self.profile_larger_than_psf_col = "profile_larger_than_psf"

        self.required_input_columns = ["cd_bg_result_dataclass"]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.profile_larger_than_psf_col] = df.apply(
            lambda row: column_density_larger_than_psf_threshold(
                cd_bg=row.cd_bg_result_dataclass,
                num_psfs_required=self.num_psfs_required,
            ),
            axis=1,
        )
        return df


class CheckVectorialFittingReliable(EpochPostProcessingStep):
    def __init__(self):
        super().__init__()
        self.vectorial_fitting_reliable_col = "vectorial_fitting_reliable"

        self.required_input_columns = [
            "vectorial_fitting_spatial_coverage_satisfied",
            "profile_larger_than_psf",
        ]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.vectorial_fitting_reliable_col] = (
            df.vectorial_fitting_spatial_coverage_satisfied & df.profile_larger_than_psf
        )
        return df


class ExpandBackgroundAnalysisColumns(EpochPostProcessingStep):
    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df[
            ["last_usable_r_km", "last_usable_cd_per_cm2", "background_oh_cd_per_cm2"]
        ] = df.apply(
            lambda row: pd.Series(
                {
                    "last_usable_r_km": row.last_usable_r.to_value(u.km),  # type: ignore
                    "last_usable_cd_per_cm2": row.last_usable_cd.to_value(1 / u.cm**2),  # type: ignore
                    "background_oh_cd_per_cm2": row.background_oh_cd.to_value(
                        1 / u.cm**2  # type: ignore
                    ),
                }
            ),
            axis=1,
        )
        df = df.drop(columns=["last_usable_r", "last_usable_cd", "background_oh_cd"])
        return df


class CleanUpTemporaryColumns(EpochPostProcessingStep):
    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        columns_to_delete = ["cd_bg_result_dataclass"]
        df = df_in.copy()
        df = df.drop(columns=columns_to_delete).reset_index(drop=True)
        return df


def do_vectorial_fitting_reliability_post_processing(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    dust_rednesses: list[DustReddeningPercent],
    vectorial_fitting_requires_km: float,
    num_psfs_required: float,
) -> pd.DataFrame | None:
    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    initial_dataframe = pd.DataFrame({})
    vec_fit_test_pipeline_steps = [
        CreateDataframeFromEpochSummary(epoch_summary=epoch_summary),
        AddStackingMethodStep(stacking_method=stacking_method),
        AddDustRednessesStep(dust_rednesses=dust_rednesses),
        ColumnDensityAboveBackgroundAnalysisStep(scp=scp),
        CheckSufficientColumnDensityCoverage(
            vectorial_fitting_requires_km=vectorial_fitting_requires_km
        ),
        CheckProfileExtendsBeyondPSF(num_psfs_required=num_psfs_required),
        CheckVectorialFittingReliable(),
        ExpandBackgroundAnalysisColumns(),
        CleanUpTemporaryColumns(),
    ]

    # TODO: enable post-process cache
    results_cache_path = (
        pathlib.Path(scp.pipeline_files.base_project_path)
        / "post_processing"
        / f"vectorial_fitting_reliability_{epoch_id}_{stacking_method}.csv"
    )

    df = apply_epoch_post_processing_pipeline(
        initial_dataframe=initial_dataframe,
        ep=vec_fit_test_pipeline_steps,
        # results_cache_path=results_cache_path,
    )

    return df
