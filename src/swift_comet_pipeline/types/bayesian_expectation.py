from dataclasses import dataclass


@dataclass
class BayesianExpectationResultFromDataframe:
    domain_column: str
    value_column: str
    expectation_value: float
