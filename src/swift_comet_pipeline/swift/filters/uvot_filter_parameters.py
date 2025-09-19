from swift_comet_pipeline.scp_types.primitive import *


_filter_params = {
    UvotFilter.uvv: UvotFilterParameters(
        fwhm=769,
        zero_point=17.89,
        zero_point_err=0.013,
        calibrated_flux=2.61e-16,
        calibrated_flux_err=2.4e-18,
    ),
    UvotFilter.ubb: UvotFilterParameters(
        fwhm=975,
        zero_point=19.11,
        zero_point_err=0.016,
        calibrated_flux=1.32e-16,
        calibrated_flux_err=9.2e-18,
    ),
    UvotFilter.uuu: UvotFilterParameters(
        fwhm=785,
        zero_point=18.34,
        zero_point_err=0.020,
        calibrated_flux=1.5e-16,
        calibrated_flux_err=1.4e-17,
    ),
    UvotFilter.uw1: UvotFilterParameters(
        fwhm=693,
        zero_point=17.49,
        zero_point_err=0.03,
        calibrated_flux=4.3e-16,
        calibrated_flux_err=2.1e-17,
    ),
    UvotFilter.um2: UvotFilterParameters(
        fwhm=498,
        zero_point=16.82,
        zero_point_err=0.03,
        calibrated_flux=7.5e-16,
        calibrated_flux_err=1.1e-17,
    ),
    UvotFilter.uw2: UvotFilterParameters(
        fwhm=657,
        zero_point=17.35,
        zero_point_err=0.04,
        calibrated_flux=6.0e-16,
        calibrated_flux_err=6.4e-17,
    ),
}


# TODO: these are all technically a function of time, so we should incorporate that as an entry in the dataclass
# and this function call
def get_filter_parameters(filter_type: UvotFilter) -> UvotFilterParameters | None:
    return _filter_params.get(filter_type, None)
