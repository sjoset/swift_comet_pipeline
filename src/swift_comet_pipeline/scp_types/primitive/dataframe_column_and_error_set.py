from dataclasses import dataclass


@dataclass
class DataframeColumnAndErrorSet:
    """Strings describing the column names of a value, its variance, and error"""

    col: str
    col_var: str
    col_err: str
