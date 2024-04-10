# from typing import TypeAlias
#
# from scipy.interpolate import interp1d
#
# from swift_comet_pipeline.error.error_propogation import ValueAndStandardDev
# from swift_comet_pipeline.production.fluorescence_OH import NumOH
#
#
# NumQH2O: TypeAlias = ValueAndStandardDev
#
#
# def num_OH_to_Q_vectorial(helio_r_au: float, num_OH: NumOH) -> NumQH2O:
#     # TODO: this is incredibly to ugly to hard-code vm_Q here
#
#     # dummy water production that the models were run with
#     vm_Q = 1.0e28
#
#     spc = read_swift_pipeline_config()
#     if spc is None:
#         print("Could not load pipeline config")
#         return NumQH2O(value=0, sigma=0)
#
#     vectorial_model_path = spc.vectorial_model_path
#     vm_df = pd.read_csv(vectorial_model_path)
#
#     r_vs_num_OH_interp = interp1d(
#         vm_df["r (AU)"], vm_df["total_fragment_number"], fill_value="extrapolate"  # type: ignore
#     )
#
#     predicted_num_OH = r_vs_num_OH_interp(helio_r_au)
#
#     predicted_to_actual = predicted_num_OH / num_OH.value
#
#     q = vm_Q / predicted_to_actual
#     q_err = (vm_Q / predicted_num_OH) * num_OH.sigma
#
#     return NumQH2O(value=q, sigma=q_err)
