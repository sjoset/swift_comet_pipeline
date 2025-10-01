import numpy as np
from astropy.nddata import block_reduce

from swift_comet_pipeline.scp_types.primitive.swift_uvot_image import SwiftUvotImage


def downsample_event_mode_image(img: SwiftUvotImage):
    """
    Takes an event mode image and reduces pixel resolution via downsampling to produce 1 arcsecond resolution image.
    The middle of the incoming image will be the middle of the outgoing image - if an image is centered on the comet nucleus,
    this will preserve that property.

    Assumes that the incoming image is odd dimensions in width and height.
    """

    # TODO: check if image has odd dimensions and complain

    h, w = img.shape

    pad_h = h % 2
    pad_w = w % 2

    # Pad with zeros at the edges
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    reduced = block_reduce(padded, block_size=(2, 2), func=np.sum)

    return reduced
