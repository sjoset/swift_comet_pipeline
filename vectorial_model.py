import pathlib
import pandas as pd

from scipy.interpolate import interp1d


__version__ = "0.0.1"

__all__ = ["num_OH_to_Q"]


def num_OH_to_Q(
    helio_r: float, num_OH: float, vectorial_model_path: pathlib.Path
) -> float:
    # TODO: this is incredibly to ugly to hard-code this vm_Q here
    # dummy production the models were run with
    vm_Q = 1.0e28
    vm_df = pd.read_csv(vectorial_model_path)

    r_vs_num_OH_interp = interp1d(
        vm_df["r (AU)"], vm_df["total_fragment_number"], fill_value="extrapolate"  # type: ignore
    )

    predicted_num_OH = r_vs_num_OH_interp(helio_r)

    predicted_to_actual = predicted_num_OH / num_OH

    return vm_Q / predicted_to_actual
