from functools import cache

# import numpy as np
import astropy.units as u

from swift_comet_pipeline.swift.effective_wavelength import (
    effective_wavelength_of_filter,
)

# from swift_comet_pipeline.swift.read_filter_effective_area import (
#     read_filter_effective_area,
# )
from swift_comet_pipeline.types import DustReddeningPercent, SwiftFilter


__all__ = ["reddening_correction"]


@cache
def reddening_correction(dust_redness: DustReddeningPercent) -> float:
    """
    get the correction factor of beta for dust reddening
    units of reddening: %/100nm

    where beta is the factor in (uw1 - beta * uvv)

    TODO: document derivation in thesis, including explanation of magic number 20000 % per nm
    """

    l_uvw1 = effective_wavelength_of_filter(SwiftFilter.uw1)
    l_uvv = effective_wavelength_of_filter(SwiftFilter.uvv)

    dlambda_nm = (l_uvv - l_uvw1).to_value(u.nm)  # type: ignore
    t = dust_redness * dlambda_nm / 20000

    return (1 - t) / (1 + t)
