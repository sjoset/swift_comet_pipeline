import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    LogStretch,
    # ZScaleInterval,
)

from swift_comet_pipeline.types.pixel_coord import PixelCoord


def plot_images_multi(images: list, comet_centers: list[PixelCoord] | None):

    # zscale = ZScaleInterval()

    # scale_limits = [zscale.get_limits(images[0]), zscale.get_limits(images[0]), zscale.get_limits(images[2]), zscale.get_limits(images[2]), zscale.get_limits(images[4]), zscale.get_limits(images[4])]

    num_images = len(images)

    _, axs = plt.subplots(1, num_images, figsize=(12 * num_images, 12))

    if len(images) == 1:
        axs = [axs]

    if comet_centers is None:
        comet_centers = [
            PixelCoord(x=np.floor(img.shape[1] / 2), y=np.floor(img.shape[0] / 2))
            for img in images
        ]

    for ax, img, comet_center in zip(axs, images, comet_centers):

        # comet_coords = PixelCoord( x=np.floor(img.shape[1] / 2), y=np.floor(img.shape[0] / 2))
        norm = ImageNormalize(
            img, interval=PercentileInterval(99.5), stretch=LogStretch()  # type: ignore
        )
        # vmin, vmax = zscale.get_limits(img)

        # vmin, vmax = sl
        # ax.imshow(img, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)

        ax.imshow(img, origin="lower", cmap="viridis", norm=norm)

        ax.axvline(comet_center.x, color="b", alpha=0.15)
        ax.axhline(comet_center.y, color="b", alpha=0.15)

    plt.show()
    plt.close()
