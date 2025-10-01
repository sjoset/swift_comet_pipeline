from dataclasses import dataclass

from astropy.io import fits

from swift_comet_pipeline.scp_types.primitive import *


@dataclass
class StackableUvotImagePrecursor:
    """
    Read directly from raw FITS data - event mode images need an extra time-slice binning step to become stackable
    """

    horizons_id: str
    img_hdr: fits.Header
    img: SwiftUvotImage | fits.FITS_rec
    comet_center: PixelCoord
    exposure_time_s: float
    data_mode: UvotImageMode


@dataclass
class StackableUvotImage:
    """
    Coincidence-corrected images with all below data, for inclusion in sum and median stacks
    """

    img: SwiftUvotImage
    comet_center: PixelCoord
    exposure_time_s: float
    data_mode: UvotImageMode


@dataclass
class EventModeTimeBinImageResult:
    """
    Resulting data from time-binning a raw event-mode image and stacking.
    """

    sum: StackableUvotImage
    median: StackableUvotImage
    exposure_map: StackableUvotImage
