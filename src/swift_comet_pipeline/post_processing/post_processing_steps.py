import pathlib
from dataclasses import asdict
from itertools import accumulate
from typing import TypeAlias
from functools import reduce

import pandas as pd

from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.epoch_summary import EpochSummary
from swift_comet_pipeline.types.stacking_method import StackingMethod


# TODO: these probably belong somewhere else


# base class for post-processing steps that operate epoch-by-epoch
class EpochPostProcessingStep:
    def __init__(self):
        self.required_input_columns = None

    def __call__(self, _: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Each PostProcessingStep must define a __call__ method."
        )

    def check_required_columns(self, df_in: pd.DataFrame, step_name: str) -> bool:
        if self.required_input_columns is None:
            return True
        if not all([x in df_in.columns for x in self.required_input_columns]):  # type: ignore
            print(f"Not all required columns for {step_name} present!")
            print(f"Present: {df_in.columns=}")
            print(f"Need: {self.required_input_columns=}")
            return False
        return True


EpochPostProcessingPipeline: TypeAlias = list[EpochPostProcessingStep]


# adds the entries of the EpochSummary data structure as columns to a dataframe
class AddEpochSummary(EpochPostProcessingStep):
    def __init__(self, epoch_summary: EpochSummary):
        self.epoch_summary = epoch_summary

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df = df.assign(**asdict(self.epoch_summary))
        return df


# creates a dataframe from epoch summary instead of adding columns to existing dataframe
class CreateDataframeFromEpochSummary(EpochPostProcessingStep):
    def __init__(self, epoch_summary: EpochSummary):
        self.epoch_summary = epoch_summary

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()

        epoch_summary_dict = asdict(self.epoch_summary)
        df = pd.concat([df, pd.DataFrame([epoch_summary_dict])], ignore_index=True)
        return df


# adds column to identify the stacking method used for analysis
class AddStackingMethodStep(EpochPostProcessingStep):
    def __init__(
        self,
        stacking_method: StackingMethod,
    ):
        self.stacking_method = stacking_method
        self.stacking_method_column = "stacking_method"

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df[self.stacking_method_column] = self.stacking_method
        return df


# add each dust redness in the list as a column of its own
class AddDustRednessesStep(EpochPostProcessingStep):
    def __init__(
        self,
        dust_rednesses: list[DustReddeningPercent],
    ):
        self.dust_rednesses = dust_rednesses
        self.redness_column = "dust_redness"

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df[self.redness_column] = [self.dust_rednesses]
        df = df.explode(self.redness_column).reset_index(drop=True)
        return df


def apply_epoch_post_processing_pipeline(
    initial_dataframe: pd.DataFrame,
    ep: EpochPostProcessingPipeline,
    results_cache_path: pathlib.Path | None = None,
) -> pd.DataFrame | None:

    if results_cache_path is None:
        df = reduce(lambda df, pipe_step: pipe_step(df), ep, initial_dataframe)
        return df

    if results_cache_path.exists():
        # TODO: this should be logged instead of printed
        # print(f"Returning results stored in {results_cache_path} ...")
        df = pd.read_csv(results_cache_path)
        return df

    df = reduce(lambda df, pipe_step: pipe_step(df), ep, initial_dataframe)
    results_cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_cache_path)
    return df


# apply the given pipeline steps, but return a list of
# [initial_dataframe, dataframe_after_step_1, ..., final_dataframe]
def apply_epoch_post_processing_pipeline_accumulate(
    initial_dataframe: pd.DataFrame, ep: EpochPostProcessingPipeline
) -> list[pd.DataFrame]:

    # df = reduce(lambda df, pipe_step: pipe_step(df), ep, initial_dataframe)
    df_list = list(
        accumulate(ep, lambda df, pipe_step: pipe_step(df), initial=initial_dataframe)
    )

    return df_list
