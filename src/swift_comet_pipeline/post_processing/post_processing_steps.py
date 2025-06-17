import pathlib
from dataclasses import asdict
from itertools import accumulate
from typing import TypeAlias
from functools import reduce

import pandas as pd

from swift_comet_pipeline.types.epoch_summary import EpochSummary


# generic pipeline step
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


class AddEpochSummary(EpochPostProcessingStep):
    def __init__(self, epoch_summary: EpochSummary):
        self.epoch_summary = epoch_summary

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df = df.assign(**asdict(self.epoch_summary))
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
