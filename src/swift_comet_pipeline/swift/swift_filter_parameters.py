from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_filter_parameters import SwiftFilterParameters


_filter_params = {
    SwiftFilter.uvv: SwiftFilterParameters(
        fwhm=769,
        zero_point=17.89,
        zero_point_err=0.013,
        calibrated_flux=2.61e-16,
        calibrated_flux_err=2.4e-18,
    ),
    SwiftFilter.ubb: SwiftFilterParameters(
        fwhm=975,
        zero_point=19.11,
        zero_point_err=0.016,
        calibrated_flux=1.32e-16,
        calibrated_flux_err=9.2e-18,
    ),
    SwiftFilter.uuu: SwiftFilterParameters(
        fwhm=785,
        zero_point=18.34,
        zero_point_err=0.020,
        calibrated_flux=1.5e-16,
        calibrated_flux_err=1.4e-17,
    ),
    SwiftFilter.uw1: SwiftFilterParameters(
        fwhm=693,
        zero_point=17.49,
        zero_point_err=0.03,
        calibrated_flux=4.3e-16,
        calibrated_flux_err=2.1e-17,
    ),
    SwiftFilter.um2: SwiftFilterParameters(
        fwhm=498,
        zero_point=16.82,
        zero_point_err=0.03,
        calibrated_flux=7.5e-16,
        calibrated_flux_err=1.1e-17,
    ),
    SwiftFilter.uw2: SwiftFilterParameters(
        fwhm=657,
        zero_point=17.35,
        zero_point_err=0.04,
        calibrated_flux=6.0e-16,
        calibrated_flux_err=6.4e-17,
    ),
}


# TODO: these are all technically a function of time, so we should incorporate that as an entry in the dataclass
# and this function call
def get_filter_parameters(filter_type: SwiftFilter) -> SwiftFilterParameters | None:
    return _filter_params.get(filter_type, None)


# def get_filter_parameters(filter_type: SwiftFilter) -> Dict:
#     filter_params = {
#         SwiftFilter.uvv: {
#             "fwhm": 769,
#             "zero_point": 17.89,
#             "zero_point_err": 0.013,
#             "cf": 2.61e-16,
#             "cf_err": 2.4e-18,
#         },
#         SwiftFilter.ubb: {
#             "fwhm": 975,
#             "zero_point": 19.11,
#             "zero_point_err": 0.016,
#             "cf": 1.32e-16,
#             "cf_err": 9.2e-18,
#         },
#         SwiftFilter.uuu: {
#             "fwhm": 785,
#             "zero_point": 18.34,
#             "zero_point_err": 0.020,
#             "cf": 1.5e-16,
#             "cf_err": 1.4e-17,
#         },
#         SwiftFilter.uw1: {
#             "fwhm": 693,
#             "zero_point": 17.49,
#             "zero_point_err": 0.03,
#             "cf": 4.3e-16,
#             "cf_err": 2.1e-17,
#         },
#         SwiftFilter.um2: {
#             "fwhm": 498,
#             "zero_point": 16.82,
#             "zero_point_err": 0.03,
#             "cf": 7.5e-16,
#             "cf_err": 1.1e-17,
#         },
#         SwiftFilter.uw2: {
#             "fwhm": 657,
#             "zero_point": 17.35,
#             "zero_point_err": 0.04,
#             "cf": 6.0e-16,
#             "cf_err": 6.4e-17,
#         },
#     }
#     return filter_params[filter_type]
