from typing import Callable

import numpy as np
import pandas as pd

from swift_comet_pipeline.scp_types.primitive.bayesian_expectation import (
    BayesianExpectationResultFromDataframe,
    PhysicalBayesianExpectationValues,
)


def bayesian_expectation_over_distribution(
    df: pd.DataFrame, domain_column: str, value_columns: list[str], pdf
) -> list[BayesianExpectationResultFromDataframe]:
    """
    Performs integral/discrete sum of pdf(domain_column) * value_column

    The values in domain_column might not cover the entire domain of pdf(x), so
    the pdf is normalized over the values we use from the domain_column.
    No checks are made to ensure that the normalization sum of pdf(x) over
    domain_column is not zero!
    """

    x = df[domain_column].values
    w_raw = [pdf(y) for y in x]
    w_norm = w_raw / np.sum(w_raw)

    ers = [
        BayesianExpectationResultFromDataframe(
            domain_column=domain_column,
            value_column=vc,
            expectation_value=np.sum(df[vc].values * w_norm),
        )
        for vc in value_columns
    ]
    return ers


def bayesian_expectation_over_distribution_physical_only(
    df: pd.DataFrame,
    domain_column: str,
    value_columns: list[str],
    pdf: Callable[[np.ndarray], np.ndarray],
    physical_lambda: Callable[[pd.DataFrame], pd.Series],
) -> PhysicalBayesianExpectationValues:
    """
    Performs integral/discrete sum of pdf(domain_column) * value_column

    The values in domain_column might not cover the entire domain of pdf(x), so
    the pdf is normalized over the values we use from the domain_column.
    No checks are made to ensure that the normalization sum of pdf(x) over
    domain_column is not zero!
    """

    xs = df[domain_column].to_numpy()
    w_raw = pdf(xs)
    total_probability = np.sum(w_raw)

    # TODO:
    # if total_probability <= 0:

    only_physical_mask = physical_lambda(df).to_numpy()

    prob_nonphysical = float(w_raw[~only_physical_mask].sum())
    percent_nonphysical = 100 * prob_nonphysical / total_probability

    w_phys_raw = w_raw[only_physical_mask]
    prob_physical = float(w_phys_raw.sum())

    # TODO:
    # if prob_physical <= 0:

    # re-normalized probability weights over the physically-valid values
    w_phys = w_phys_raw / prob_physical

    expectation_values: dict[str, float] = {}

    for val_col in value_columns:
        vs = df.loc[only_physical_mask, val_col].to_numpy()
        expectation_values[val_col] = float(np.sum(w_phys * vs))

    return PhysicalBayesianExpectationValues(
        total_probability=total_probability,
        nonphysical_probability=prob_nonphysical,
        percent_nonphysical=percent_nonphysical,
        expectations=expectation_values,
    )
