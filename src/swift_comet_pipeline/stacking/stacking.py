import logging as log

from astropy.time import Time
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import EpochSubpipelineKey
from swift_comet_pipeline.scp_types.primitive import *

from swift_comet_pipeline.image_manipulation.event_mode_downsample import (
    downsample_event_mode_image,
)
from swift_comet_pipeline.image_manipulation.get_stacked_uvot_image_center import (
    get_uvot_image_center,
)
from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
)
from swift_comet_pipeline.data_ingestion.observation_log.comet_center_tracking import (
    get_comet_center_prefer_user_coords,
)
from swift_comet_pipeline.data_ingestion.epoch_index.find_epoch_index_entry import (
    get_epoch_index_entry,
)
from swift_comet_pipeline.scp_types.compound.stacking import (
    StackableUvotImage,
    StackableUvotImagePrecursor,
)
from swift_comet_pipeline.scp_types.primitive.swift_uvot_observation_log_dataframe import (
    SwiftUvotObservationLogDataframe,
)
from swift_comet_pipeline.stacking.determine_stack_size import (
    determine_stacking_image_size_from_stackables,
)
from swift_comet_pipeline.stacking.event_mode import (
    event_mode_fits_to_time_binned_image,
)
from swift_comet_pipeline.stacking.stacked_image_to_fits_image_hdu import (
    stacked_image_to_fits,
)
from swift_comet_pipeline.swift.coincidence_correct.coincidence_correct_map import (
    coincidence_correction_factor_map,
)
from swift_comet_pipeline.swift.uvot_sensitivity import (
    uvot_sensitivity_correction_factor,
)


def epoch_row_to_stacking_precursor(
    row: pd.Series, horizons_id: str
) -> StackableUvotImagePrecursor:

    comet_center = get_comet_center_prefer_user_coords(row=row)
    exposure_time_s = row.EXPOSURE
    data_mode = row.DATAMODE
    img = fits.getdata(filename=row.FULL_FITS_PATH, ext=row.EXTENSION)
    assert img is not None
    hdr = fits.getheader(filename=row.FULL_FITS_PATH, ext=row.EXTENSION)
    assert hdr is not None

    precursor = StackableUvotImagePrecursor(
        img_hdr=hdr,
        img=img,  # type: ignore
        comet_center=comet_center,
        exposure_time_s=exposure_time_s,
        data_mode=data_mode,
        horizons_id=horizons_id,
    )
    return precursor


def process_stackable_precursor(
    precursor: StackableUvotImagePrecursor, do_coincidence_correction: bool
) -> StackableUvotImage:
    """
    For data mode images, applies coincidence correction

    For event mode images: bin by time, coincidence correct, and stack slices
    """

    if precursor.data_mode == UvotImageMode.data_mode:
        assert isinstance(precursor.img, SwiftUvotImage)
        if do_coincidence_correction:
            coi_map = coincidence_correction_factor_map(
                img=precursor.img, scale=UvotPixelResolution.data_mode
            )
            cc_img = coi_map * precursor.img
        else:
            cc_img = precursor.img
        stackable = StackableUvotImage(
            img=cc_img,
            comet_center=precursor.comet_center,
            exposure_time_s=precursor.exposure_time_s,
            data_mode=UvotImageMode.data_mode,
        )
    else:
        # TODO: make num_time_slices an option in the user config, or some way to control the maximum time slice length
        # TODO: we throw away the exposure mask from the event mode stack - probably fine as the offsets between sub-slices are small
        num_time_slices = int(np.ceil(precursor.exposure_time_s / 30.0))
        binning_result = event_mode_fits_to_time_binned_image(
            precursor_img=precursor,
            num_time_slices=num_time_slices,
            do_coincidence_correction=do_coincidence_correction,
        )
        stackable = binning_result.sum

    return stackable


def downsample_event_mode_stackable_image(s: StackableUvotImage) -> StackableUvotImage:
    """
    Requires an event mode image to be centered on the comet
    """

    # this downsampling preserves the centering on the comet
    downsampled_img = downsample_event_mode_image(img=s.img)

    downsampled_stackable = StackableUvotImage(
        img=downsampled_img,
        comet_center=get_uvot_image_center(downsampled_img),
        exposure_time_s=s.exposure_time_s,
        data_mode=UvotImageMode.data_mode,
    )

    return downsampled_stackable


def uniform_pixel_resolution(
    imgs: list[StackableUvotImage],
) -> list[StackableUvotImage]:
    """
    Take a list and down-sample all event-mode images so that every image has pixel scale of 1 arcsecond
    """

    new_imgs = []
    for i in imgs:
        if i.data_mode == UvotImageMode.data_mode:
            new_imgs.append(i)
        else:
            new_imgs.append(downsample_event_mode_stackable_image(i))

    return new_imgs


def stack_images(
    stackable_images: list[StackableUvotImage],
    stacking_image_final_size_rows_cols: tuple[int, int],
) -> tuple[SwiftUvotImage, SwiftUvotImage, SwiftUvotImage] | None:

    if len(stackable_images) == 0:
        return None

    # TODO: selectively exclude comets if they fall outside the image bounds!

    # # check if the comet center is outside the bounds of the image and omit it
    # img_height, img_width = image_data.shape  # type: ignore
    # if comet_center_coords.x < 0 or comet_center_coords.x > img_width:
    #     print(f"Image dimensions ==> width={img_width}\theight={img_height}")
    #     print(f"Invalid comet x coordinate {comet_center_coords.x}! Skipping.")
    #     continue
    # if comet_center_coords.y < 0 or comet_center_coords.y > img_height:
    #     print(f"Image dimensions ==> width={img_width}\theight={img_height}")
    #     print(f"Invalid comet y coordinate {comet_center_coords.y}! Skipping.")
    #     continue

    final_img_size = stacking_image_final_size_rows_cols

    print("Resizing images ...  ", end="")
    resized_images_to_stack = [
        center_image_on_coords(
            s.img,
            source_coords_to_center=s.comet_center,
            stacking_image_size=final_img_size,
            # show_resulting_image=True,
        )
        for s in stackable_images
    ]

    exposure_times = [s.exposure_time_s for s in stackable_images]

    print("Calculating exposure map ...  ", end="")
    exposure_map_list = []
    for resized_img, exp_time in zip(resized_images_to_stack, exposure_times):
        dead_pixels = resized_img == 0
        good_pix = np.ones_like(resized_img) * exp_time
        good_pix[dead_pixels] = 0
        exposure_map_list.append(good_pix)

    final_exposure_map = np.sum(exposure_map_list, axis=0)
    total_exposure_time_s = np.sum(exposure_times)

    print("Calculating sum stacks ...  ", end="")
    stack_sum = np.sum(resized_images_to_stack, axis=0) / total_exposure_time_s

    print("Calculating median stacks ...  ", end="")
    stack_median = np.median(
        [
            img / exp_time_s
            for img, exp_time_s in zip(resized_images_to_stack, exposure_times)
        ],
        axis=0,
    )

    return stack_sum, stack_median, final_exposure_map


def sum_median_and_exposure_map(
    filtered_obs_log: SwiftUvotObservationLogDataframe,
    filter_type: UvotFilter,
    observation_mid_time: pd.Timestamp,
    horizons_id: str,
    do_coincidence_correction: bool,
    do_sensitivity_correction: bool,
) -> tuple[SwiftUvotImage, SwiftUvotImage, SwiftUvotImage] | None:
    """
    Blindly takes every entry in the given log and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    If successful, returns a tuple of images: (sum, median, exposure_map)
    The exposure_map image has pixels with values in units of seconds - the total exposure time from the stack of images involved
    """

    event_mode_epoch = filtered_obs_log[
        filtered_obs_log.DATAMODE == UvotImageMode.event_mode
    ].copy()
    data_mode_epoch = filtered_obs_log[
        filtered_obs_log.DATAMODE == UvotImageMode.data_mode
    ].copy()

    print(
        f"Event mode images: {len(event_mode_epoch)}\t\tData mode images: {len(data_mode_epoch)}"
    )

    print("Creating precursors ...  ", end="")
    stacking_precursors = [
        epoch_row_to_stacking_precursor(row=row, horizons_id=horizons_id)
        for _, row in tqdm(
            filtered_obs_log.iterrows(), total=len(filtered_obs_log), unit="images"
        )
    ]

    print("Processing precursors ...")
    # this can take a while - coincidence correct, and for event mode - time slice and stack also
    stackable_images: list[StackableUvotImage] = []
    for sp in tqdm(stacking_precursors, total=len(stacking_precursors), unit="images"):
        stackable_images.append(
            process_stackable_precursor(
                sp, do_coincidence_correction=do_coincidence_correction
            )
        )

    # print("Processing precursors ...")
    # process_one = partial(
    #     process_stackable_precursor,
    #     do_coincidence_correction=do_coincidence_correction,
    # )
    # with ProcessPoolExecutor() as ex:
    #     stackable_images: list[StackableUVOTImage] = list(
    #         tqdm(
    #             ex.map(process_one, stacking_precursors),  # preserves order
    #             total=len(stacking_precursors),
    #             unit="images",
    #         )
    #     )

    print("Applying uniform resolution sampling ...  ", end="")
    stackable_images = uniform_pixel_resolution(stackable_images)

    print("Determining final stacked image size ...  ", end="")
    stacking_image_size = determine_stacking_image_size_from_stackables(
        stackable_images
    )
    print("Done ... ", end="")

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    stack_results = stack_images(
        stackable_images=stackable_images,
        stacking_image_final_size_rows_cols=stacking_image_size,
    )

    if stack_results is None:
        print("Could not finalize stack! Not stacking.")
        return None

    if do_sensitivity_correction:
        uvot_correction_factor = (
            uvot_sensitivity_correction_factor(
                filter_type=filter_type, t_obs=Time(observation_mid_time)
            )
            or 1.0
        )
        print(
            f"\nApplying UVOT sensitivity corrections for {observation_mid_time} with factor {uvot_correction_factor:3.2f} ...  ",
            end="",
        )
    else:
        print("No sensitivity correction requested.")
        uvot_correction_factor = 1.0

    # Adjust the sum and median, leave the exposure map alone
    sensitivity_corrected = (
        stack_results[0] * uvot_correction_factor,
        stack_results[1] * uvot_correction_factor,
        stack_results[2],
    )
    print("Complete!")

    return sensitivity_corrected


# should return StackedImageResultSet: sum, median, exposuremap
# the caller should then update all products: if one changes, they should all change - if we only wanted sum, too bad - we get the median with very little extra work
# computing sum and median separately would be a massive slowdown
def do_stacking(
    scp: Products,
    key: EpochSubpipelineKey,
    do_coincidence_correction: bool = True,
    do_sensitivity_correction: bool = True,
) -> tuple[fits.ImageHDU, fits.ImageHDU, fits.ImageHDU] | None:
    """
    TODO: update description
    """

    obs_log = scp.load_obs_log()
    epoch_index = scp.load_epoch_index()

    if obs_log is None or epoch_index is None:
        log.info(
            f"Error loading observation log or epoch index during stacking of {key}"
        )
        return

    epoch_index_entry = get_epoch_index_entry(
        epoch_index=epoch_index, epoch_id=key.epoch_id
    )
    if epoch_index_entry is None:
        return None

    obs_log_mask = (
        (obs_log.epoch_id == key.epoch_id)
        & (obs_log.FILTER == key.filter_type)
        & (obs_log.manual_veto == np.False_)
    )
    filtered_obs_log: SwiftUvotObservationLogDataframe = obs_log[obs_log_mask]  # type: ignore

    if len(filtered_obs_log) == 0:
        log.info(f"No observations for {key} found in log")
        return

    sum_median_exposure = sum_median_and_exposure_map(
        filtered_obs_log=filtered_obs_log,
        filter_type=key.filter_type,
        observation_mid_time=epoch_index_entry.observation_time,
        horizons_id=scp.cfg.jpl_horizons_id,
        do_coincidence_correction=do_coincidence_correction,
        do_sensitivity_correction=do_sensitivity_correction,
    )
    if sum_median_exposure is None:
        log.info(f"Failed to build stacked image for {key}")
        return None

    img_sum, img_median, img_exp_map = sum_median_exposure

    fits_sum = stacked_image_to_fits(
        epoch_index_entry=epoch_index_entry, img=img_sum, filter_type=key.filter_type
    )
    fits_med = stacked_image_to_fits(
        epoch_index_entry=epoch_index_entry, img=img_median, filter_type=key.filter_type
    )
    fits_exp = stacked_image_to_fits(
        epoch_index_entry=epoch_index_entry,
        img=img_exp_map,
        filter_type=key.filter_type,
    )

    return (fits_sum, fits_med, fits_exp)
