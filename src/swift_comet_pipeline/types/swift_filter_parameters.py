from dataclasses import dataclass


# TODO: cite these from the swift documentation
# TODO: these are all technically a function of time, so we should incorporate that as an entry in the dataclass
@dataclass
class SwiftFilterParameters:
    fwhm: float
    zero_point: float
    zero_point_err: float
    # calibrated to Vega
    calibrated_flux: float
    calibrated_flux_err: float
