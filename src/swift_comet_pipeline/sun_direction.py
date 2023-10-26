import numpy as np
import matplotlib.pyplot as plt

from astropy.visualization import ZScaleInterval
from astropy.coordinates import get_sun
from astropy.time import Time
from astropy.wcs.utils import skycoord_to_pixel

from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.epochs import read_epoch
from swift_comet_pipeline.tui import epoch_menu

__all__ = ["find_sun_direction", "show_fits_sun_direction"]


""" Untested and probably doesn't work correctly """


def show_fits_sun_direction(img, sun_x, sun_y, comet_x, comet_y):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img)

    im1 = ax1.imshow(img, vmin=vmin, vmax=vmax)  # type: ignore
    fig.colorbar(im1)

    ax1.plot([sun_x, comet_x], [sun_y, comet_y])  # type: ignore
    ax1.set_xlim(0, img.shape[1])  # type: ignore
    ax1.set_ylim(0, img.shape[0])  # type: ignore
    # ax1.axvline(image_center_col, color="b", alpha=0.2)
    # ax1.axhline(image_center_row, color="b", alpha=0.2)

    plt.show()


def find_sun_direction(swift_data: SwiftData, pipeline_files: PipelineFiles) -> None:
    """Selects an epoch and attempts to find the direction of the sun on the raw images"""
    epoch_prod = epoch_menu(pipeline_files=pipeline_files)
    if epoch_prod is None:
        return
    epoch_path = epoch_prod.product_path
    epoch_pre_veto = read_epoch(epoch_path)

    filter_mask = epoch_pre_veto["FILTER"] == SwiftFilter.uvv
    epoch_pre_veto = epoch_pre_veto[filter_mask]

    for _, row in epoch_pre_veto.iterrows():
        img = swift_data.get_uvot_image(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        wcs = swift_data.get_uvot_image_wcs(
            obsid=row.OBS_ID,
            fits_filename=row.FITS_FILENAME,
            fits_extension=row.EXTENSION,
        )
        print(wcs)

        print(f"t = {Time(row.MID_TIME)}")
        sun = get_sun(Time(row.MID_TIME))
        print(f"{sun=}")

        sun_x, sun_y = skycoord_to_pixel(sun, wcs)
        print(sun_x, sun_y, np.degrees(np.arctan2(sun_y, sun_x)))

        show_fits_sun_direction(img, sun_x, sun_y, row.PX, row.PY)
