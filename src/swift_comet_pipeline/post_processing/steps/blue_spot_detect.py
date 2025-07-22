import pathlib

import numpy as np
import pandas as pd
import astropy.units as u

from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult
from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.comet_column_density import (
    get_comet_column_density_from_extracted_profile,
)
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.post_processing.post_processing_steps import (
    AddDustRednessesStep,
    AddStackingMethodStep,
    CreateDataframeFromEpochSummary,
    EpochPostProcessingStep,
    apply_epoch_post_processing_pipeline,
)
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod


class CalculateColumnDensityStep(EpochPostProcessingStep):
    def __init__(self, scp: SwiftCometPipeline):
        self.scp = scp

        self.required_input_columns = [
            "epoch_id",
            "rh_au",
            "stacking_method",
            "dust_redness",
        ]
        self.cd_from_profile_column = "column_density_from_profile"

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.cd_from_profile_column] = df.apply(
            lambda row: get_comet_column_density_from_extracted_profile(
                scp=self.scp,
                epoch_id=row.epoch_id,
                dust_redness=row.dust_redness,
                stacking_method=row.stacking_method,
            ),
            axis=1,
        )

        return df


class VectorialFitStep(EpochPostProcessingStep):
    def __init__(
        self,
        model_Q: u.Quantity,
        vmr: VectorialModelResult,
        near_far_radius: u.Quantity,
    ):
        self.vmr = vmr
        self.model_Q = model_Q
        self.near_far_radius = near_far_radius

        self.column_density_column = "column_density_from_profile"
        self.required_input_columns = [self.column_density_column]

        self.vectorial_fitting_column = "vectorial_fit"

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.vectorial_fitting_column] = df.apply(
            lambda row: vectorial_fit(
                comet_column_density=row.column_density_from_profile,
                model_Q=self.model_Q,
                vmr=self.vmr,
                r_fit_min=self.near_far_radius,
                r_fit_max=1.0e10 * u.km,
            ),
            axis=1,
        )

        return df


class CalculateExcessOHColumnDensityStep(EpochPostProcessingStep):
    def __init__(self):
        self.excess_oh_cd_column = "excess_oh_cd"
        self.required_input_columns = ["column_density_from_profile", "vectorial_fit"]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.excess_oh_cd_column] = df.apply(
            lambda row: row.column_density_from_profile.cd_cm2
            - row.vectorial_fit.vectorial_column_density.cd_cm2,
            axis=1,
        )

        return df


class CalculateTotalExcessOHStep(EpochPostProcessingStep):
    def __init__(self):
        self.excess_oh_column = "excess_oh"
        self.required_input_columns = ["column_density_from_profile", "excess_oh_cd"]

    def total_oh(self, row):
        rs_outer = row.column_density_from_profile.rs_km
        rs_inner = np.concatenate(([0.0], rs_outer[:-1]))

        areas_km2 = np.pi * (rs_outer**2 - rs_inner**2)
        return row.excess_oh_cd * 1e10 * areas_km2

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.excess_oh_column] = df.apply(lambda row: self.total_oh(row), axis=1)

        return df


class CalculateCumulativeExcessOHStep(EpochPostProcessingStep):
    def __init__(self):
        self.cumulative_excess_oh_column = "cumulative_excess_oh"
        self.required_input_columns = ["excess_oh"]

    def cumulative_excess_oh(self, row):
        return np.cumsum(row.excess_oh)

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.cumulative_excess_oh_column] = df.apply(
            lambda row: self.cumulative_excess_oh(row), axis=1
        )

        return df


def do_blue_spot_detection_post_processing(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    dust_rednesses: list[DustReddeningPercent],
    near_far_radius: u.Quantity,
) -> pd.DataFrame | None:
    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=epoch_summary.rh_au * u.AU)

    initial_dataframe = pd.DataFrame({})
    blue_spot_detect_pipeline_steps = [
        CreateDataframeFromEpochSummary(epoch_summary=epoch_summary),
        AddStackingMethodStep(stacking_method=stacking_method),
        AddDustRednessesStep(dust_rednesses=dust_rednesses),
        CalculateColumnDensityStep(scp=scp),
        VectorialFitStep(model_Q=model_Q, vmr=vmr, near_far_radius=near_far_radius),
        CalculateExcessOHColumnDensityStep(),
        CalculateTotalExcessOHStep(),
        CalculateCumulativeExcessOHStep(),
    ]

    # TODO: enable post-process cache
    results_cache_path = (
        pathlib.Path(scp.pipeline_files.base_project_path)
        / "post_processing"
        / f"blue_spot_detect_{epoch_id}_{stacking_method}_split_{near_far_radius.to_value(u.km)}.csv"
    )

    # TODO: our simple csv cache means our dataframe needs to be serializable/unserializable: no dataclasses etc. as columns
    df = apply_epoch_post_processing_pipeline(
        initial_dataframe=initial_dataframe,
        ep=blue_spot_detect_pipeline_steps,
        # results_cache_path=results_cache_path,
    )

    return df
