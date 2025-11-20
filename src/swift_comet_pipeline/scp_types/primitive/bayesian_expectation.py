from dataclasses import dataclass


@dataclass
class BayesianExpectationResultFromDataframe:
    """
    For blind expectation values, with no restrictions on the domain column
    """

    domain_column: str
    value_column: str
    expectation_value: float


@dataclass(frozen=True)
class PhysicalBayesianExpectationValues:
    """
    For expectation values where we want to restrict the domain column to physical values, like Q(H2O) being positive
    """

    # how much of our pdf overlaps with domain column: pdf might be zero for our domain values
    total_probability: float
    # how much of our pdf is associated with non-physical values of the domain column
    nonphysical_probability: float
    # same as nonphysical_probability, but converted to a percentage of the total_probability
    percent_nonphysical: float
    # map of column name to expectation value
    expectations: dict[str, float]
