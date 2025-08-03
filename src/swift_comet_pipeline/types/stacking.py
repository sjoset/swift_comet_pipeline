from dataclasses import dataclass

from astropy.io import fits

from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


@dataclass
class StackableUVOTImagePrecursor:
    """
    Read directly from raw FITS data - event mode images need an extra time-slice binning step to become stackable
    """

    horizons_id: str
    img_hdr: fits.Header
    img: SwiftUVOTImage | fits.FITS_rec
    comet_center: PixelCoord
    exposure_time_s: float
    data_mode: SwiftImageMode


@dataclass
class StackableUVOTImage:
    """
    Coincidence-corrected images with all below data, for inclusion in sum and median stacks
    """

    img: SwiftUVOTImage
    comet_center: PixelCoord
    exposure_time_s: float
    data_mode: SwiftImageMode


@dataclass
class EventModeTimeBinImageResult:
    """
    Resulting data from time-binning a raw event-mode image and stacking.
    """

    sum: StackableUVOTImage
    median: StackableUVOTImage
    exposure_map: StackableUVOTImage
