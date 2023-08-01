import pandas as pd
from scipy.interpolate import interp1d

from configs import read_swift_pipeline_config

__all__ = ["num_OH_to_Q_vectorial"]


def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: float) -> float:
    # TODO: this is incredibly to ugly to hard-code this vm_Q here
    # dummy production the models were run with
    vm_Q = 1.0e28

    spc = read_swift_pipeline_config()
    if spc is None:
        print("Could not load pipeline config")
        return 0
    vectorial_model_path = spc.vectorial_model_path
    vm_df = pd.read_csv(vectorial_model_path)

    r_vs_num_OH_interp = interp1d(
        vm_df["r (AU)"], vm_df["total_fragment_number"], fill_value="extrapolate"  # type: ignore
    )

    predicted_num_OH = r_vs_num_OH_interp(helio_r_au)

    predicted_to_actual = predicted_num_OH / num_OH

    return vm_Q / predicted_to_actual


# def num_OH_to_Q_vectorial(
#     helio_r_au: float,
#     num_OH: float,
#     vectorial_model_path: Optional[pathlib.Path] = None,
# ) -> float:
#     # TODO: this is incredibly to ugly to hard-code this vm_Q here
#     # dummy production the models were run with
#     vm_Q = 1.0e28
#
#     if vectorial_model_path is None:
#         spc = read_swift_pipeline_config()
#         if spc is None:
#             return 0
#         vectorial_model_path = spc.vectorial_model_path
#
#     vm_df = pd.read_csv(vectorial_model_path)
#
#     r_vs_num_OH_interp = interp1d(
#         vm_df["r (AU)"], vm_df["total_fragment_number"], fill_value="extrapolate"  # type: ignore
#     )
#
#     predicted_num_OH = r_vs_num_OH_interp(helio_r_au)
#
#     predicted_to_actual = predicted_num_OH / num_OH
#
#     return vm_Q / predicted_to_actual
