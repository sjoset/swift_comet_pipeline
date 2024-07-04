from itertools import product
from typing import TypeAlias
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage


StackedUVOTImageSet: TypeAlias = dict[
    tuple[SwiftFilter, StackingMethod], SwiftUVOTImage
]


def show_stacked_image_set(stacked_image_set: StackedUVOTImageSet, epoch_id: str = ""):
    # TODO: put this in its own file to avoid all the imports if we just want the StackedUVOTImageSet type
    _, axes = plt.subplots(2, 2, figsize=(100, 20))

    zscale = ZScaleInterval()

    filter_types = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    for i, (filter_type, stacking_method) in enumerate(
        product(filter_types, stacking_methods)
    ):
        img = stacked_image_set[(filter_type, stacking_method)]
        vmin, vmax = zscale.get_limits(img)
        ax = axes[np.unravel_index(i, axes.shape)]
        ax.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        # fig.colorbar(im)
        # TODO: get center of image from function in uvot
        ax.axvline(int(np.floor(img.shape[1] / 2)), color="b", alpha=0.1)
        ax.axhline(int(np.floor(img.shape[0] / 2)), color="b", alpha=0.1)
        ax.set_title(
            f"{epoch_id}: filter={filter_to_file_string(filter_type)}, stack type={stacking_method}"
        )

    plt.show()
    plt.close()
