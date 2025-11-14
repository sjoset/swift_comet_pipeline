import numpy as np
import pandas as pd

from swift_comet_pipeline.scp_types.primitive.bayesian_expectation import (
    BayesianExpectationResultFromDataframe,
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
