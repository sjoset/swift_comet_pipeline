import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from swift_comet_pipeline.scp_types.primitive import *


def plot_images_multi(images: list, comet_centers: list[PixelCoord] | None = None):
    """
    Returns a figure to make things like marimo happy
    """

    zscale = ZScaleInterval()
    num_images = len(images)

    fig, axs = plt.subplots(1, num_images, figsize=(12 * num_images, 12))

    if num_images == 1:
        axs = [axs]

    if comet_centers is None:
        comet_centers = [
            PixelCoord(x=np.floor(img.shape[1] / 2), y=np.floor(img.shape[0] / 2))
            for img in images
        ]

    for ax, img, comet_center in zip(axs, images, comet_centers):

        vmin, vmax = zscale.get_limits(img)

        ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)

        # norm = ImageNormalize(
        #     img, interval=PercentileInterval(99.5), stretch=LogStretch()  # type: ignore
        # )
        # ax.imshow(img, origin="lower", cmap="viridis", norm=norm)

        ax.axvline(comet_center.x, color="b", alpha=0.15)
        ax.axhline(comet_center.y, color="b", alpha=0.15)

    return fig
