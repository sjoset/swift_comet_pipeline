import pathlib
from typing import Tuple, List
from itertools import product

from astroquery.jplhorizons import Horizons
import numpy as np
from astropy.io import fits
from astropy.time import Time
from photutils.aperture import CircularAperture
from tqdm import tqdm
from icecream import ic

from swift_comet_pipeline.types.comet_center_finding_method import (
    CometCenterFindingMethod,
)
from swift_comet_pipeline.comet.comet_center import get_comet_center_prefer_user_coords
from swift_comet_pipeline.comet.comet_center_finding import find_comet_center
from swift_comet_pipeline.image_manipulation.image_pad import pad_to_match_sizes
from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
    get_image_dimensions_to_center_on_pixel,
)
from swift_comet_pipeline.observationlog.build_observation_log import (
    event_mode_header_to_WCS,
)
from swift_comet_pipeline.observationlog.epoch_typing import Epoch, EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import (
    get_unstacked_epoch_summary,
)
from swift_comet_pipeline.pipeline_utils.time_conversion import (
    uvot_time_to_astropy_time,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter_to_string import (
    filter_to_file_string,
)
from swift_comet_pipeline.observationlog.epoch import (
    epoch_stacked_image_to_fits,
    is_epoch_stackable,
)
from swift_comet_pipeline.swift.coincidence_correction import coincidence_correction
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.stacked_uvot_image_set import StackedUVOTImageSet
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def trim_image_and_relocate_pixel_coords(
    img: SwiftUVOTImage, x_min: int, x_max: int, y_min: int, y_max: int, pc: PixelCoord
) -> tuple[SwiftUVOTImage, PixelCoord]:
    """
    Trim down the given image, and move the coordinate pc based on the trim to point to the same pixel
    """

    new_img = img[y_min:y_max, x_min:x_max].copy()
    new_pixel_coord = PixelCoord(x=pc.x - x_min, y=pc.y - y_min)

    return new_img, new_pixel_coord


def get_comet_position_at_time(h_id: str, mt: Time):
    horizons_response = Horizons(
        id=h_id, location="@swift", epochs=mt.jd, id_type="designation"
    )
    eph = horizons_response.ephemerides(closest_apparition=True)  # type: ignore

    return eph["RA"][0], eph["DEC"][0]


def slice_event_mode_image_data(event_mode_bintable, x_size, y_size, num_slices):
    ts, xs, ys = (
        event_mode_bintable["TIME"],
        event_mode_bintable["X"] - 1,
        event_mode_bintable["Y"] - 1,
    )

    t_exp_start = np.min(ts)
    t_exp_stop = np.max(ts)

    slice_ts = np.linspace(t_exp_start, t_exp_stop, num=num_slices + 1, endpoint=True)
    slice_starting_ts = slice_ts[:-1]
    slice_ending_ts = slice_ts[1:]
    mid_time_list = (slice_ending_ts + slice_starting_ts) / 2

    # bump up the ending time by 1 second to catch every photon
    slice_ending_ts[-1] += 1

    image_list = []
    for t_st, t_e in zip(slice_starting_ts, slice_ending_ts):
        t_mask = np.logical_and(ts >= t_st, ts < t_e)

        img = np.zeros((y_size, x_size))
        for x, y in zip(xs[t_mask], ys[t_mask]):
            img[y, x] += 1
        image_list.append(img)

    return image_list, mid_time_list


def event_mode_fits_to_time_binned_image(
    horizons_id: str, fits_path: pathlib.Path, extension_id: int, num_time_slices: int
) -> tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage]:
    """
    Header entry TLMAX6 = maximum extent of 'X' values, but not necessarily the largest X that was observed (np.min(['X']) is lower than this value)
    Similar for TLMAX7 for the y value
    """
    ev_hdr = fits.getheader(fits_path, extension_id)
    img_wcs = event_mode_header_to_WCS(hdr=ev_hdr)
    ev_table = fits.getdata(fits_path, extension_id)
    exposure_time = float(ev_hdr["EXPOSURE"])
    exposure_time_per_slice = exposure_time / num_time_slices

    print(f"Slicing event mode table into {num_time_slices} slices ...  ", end="")
    x_min, y_min = np.min(ev_table["X"]), np.min(ev_table["Y"])
    x_max, y_max = np.max(ev_table["X"]), np.max(ev_table["Y"])

    img_slices_list, mid_time_list = slice_event_mode_image_data(
        event_mode_bintable=ev_table,
        x_size=ev_hdr["TLMAX6"],
        y_size=ev_hdr["TLMAX7"],
        num_slices=num_time_slices,
    )
    print("Done.")

    # convert times to astropy times
    astropy_mid_times = [
        uvot_time_to_astropy_time(uvot_t=t, event_mode_hdr=ev_hdr)
        for t in mid_time_list
    ]

    print("Performing Horizons lookups ...  ", end="")
    comet_ras_decs = [
        get_comet_position_at_time(h_id=horizons_id, mt=t) for t in astropy_mid_times
    ]
    slice_comet_centers = [
        img_wcs.wcs_world2pix(ra, dec, 1) for ra, dec in comet_ras_decs
    ]
    slice_comet_centers_pix = [
        PixelCoord(x=int(np.round(x)), y=int(np.round(y)))
        for x, y in slice_comet_centers
    ]
    print("Done. Comet centers during slice:")
    print(slice_comet_centers_pix)

    # trim the images down: the Xs and Ys that are measured by event mode fall within [[x_min, x_max], [y_min, y_max]]
    new_imgs_and_centers = [
        trim_image_and_relocate_pixel_coords(
            img=i, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pc=pc
        )
        for i, pc in zip(img_slices_list, slice_comet_centers_pix)
    ]

    trimmed_imgs = [x[0] for x in new_imgs_and_centers]
    trimmed_comet_centers = [x[1] for x in new_imgs_and_centers]
    print("Comet centers after trim:")
    print(trimmed_comet_centers)

    # search_aps = [CircularAperture((pc.x, pc.y), r=10) for pc in trimmed_comet_centers]
    # peak_finding_comet_centers = [
    #     find_comet_center(
    #         img=i, method=CometCenterFindingMethod.aperture_peak, search_aperture=sa
    #     )
    #     for i, sa in zip(trimmed_imgs, search_aps)
    # ]

    # don't bother with peak finding
    peak_finding_comet_centers = trimmed_comet_centers

    # see how big our final product needs to be
    event_stack_size = determine_stacking_image_size(
        img_list=trimmed_imgs, comet_center_coords=peak_finding_comet_centers
    )
    assert event_stack_size is not None
    print(f"Event mode stack size: {event_stack_size}")

    event_mode_images_to_stack = []
    for i, (trimmed_img, peak_finding_comet_center) in enumerate(
        zip(trimmed_imgs, peak_finding_comet_centers)
    ):
        print(
            f"Processing image {i+1}/{len(trimmed_imgs)}: centering on {peak_finding_comet_center} ...  ",
            end="",
        )
        image_data = center_image_on_coords(
            source_image=trimmed_img,
            source_coords_to_center=peak_finding_comet_center,
            stacking_image_size=event_stack_size,
        )
        print("coincidence correcting ...  ", end="")
        coi_map = coincidence_correction(
            img=image_data / (exposure_time_per_slice),
            scale=SwiftPixelResolution.event_mode,
        )
        image_data = image_data * coi_map
        event_mode_images_to_stack.append(image_data)
        print("Done.")

    # just, whatever idc at this point
    exposure_map = np.ones(event_stack_size)

    # divide by total exposure time so that pixels are count rates
    event_sum = np.sum(event_mode_images_to_stack, axis=0) / exposure_time

    # divide each image by its exposure time for each image to be in count rate, then take median
    event_median = np.median(
        [i / exposure_time_per_slice for i in event_mode_images_to_stack], axis=0
    )

    return event_sum, event_median, exposure_map


def determine_stacking_image_size(
    img_list: list[SwiftUVOTImage], comet_center_coords: list[PixelCoord]
) -> Tuple[int, int] | None:
    """
    Examines every image and finds the image size necessary to accommodate
    the largest image involved in the stack, so we can pad out the smaller images and stack them in one step
    """

    # stores how big each image would need to be if recentered on the comet
    recentered_image_dimensions_rows_cols = []

    recentered_image_dimensions_rows_cols = [
        get_image_dimensions_to_center_on_pixel(source_image=i, coords_to_center=pc)
        for i, pc in zip(img_list, comet_center_coords)
    ]

    if len(recentered_image_dimensions_rows_cols) == 0:
        print("No images found in epoch!")
        return None

    # now take the largest size so that every image can be stacked without losing pixels
    max_num_rows = sorted(
        recentered_image_dimensions_rows_cols, key=lambda k: k[0], reverse=True
    )[0][0]
    max_num_cols = sorted(
        recentered_image_dimensions_rows_cols, key=lambda k: k[1], reverse=True
    )[0][1]

    return (int(max_num_rows), int(max_num_cols))


# def determine_stacking_image_size(
#     swift_data: SwiftData,
#     epoch: Epoch,
# ) -> Tuple[int, int] | None:
#     """
#     Opens every FITS file specified in the given epoch and finds the image size necessary to accommodate
#     the largest image involved in the stack, so we can pad out the smaller images and stack them in one step
#     """
#
#     # stores how big each image would need to be if recentered on the comet
#     recentered_image_dimensions = []
#
#     for _, row in epoch.iterrows():
#         image_data = get_image_from_obs_log_row(swift_data=swift_data, obs_log_row=row)
#
#         comet_center_coords = get_comet_center_prefer_user_coords(row=row)
#         # keep a list of the image sizes
#         image_dimensions = get_image_dimensions_to_center_on_pixel(
#             source_image=image_data, coords_to_center=comet_center_coords
#         )
#         recentered_image_dimensions.append(image_dimensions)
#
#     if len(recentered_image_dimensions) == 0:
#         print("No images found in epoch!")
#         return None
#
#     # now take the largest size so that every image can be stacked without losing pixels
#     max_num_rows = sorted(
#         recentered_image_dimensions, key=lambda k: k[0], reverse=True
#     )[0][0]
#     max_num_cols = sorted(
#         recentered_image_dimensions, key=lambda k: k[1], reverse=True
#     )[0][1]
#
#     return (max_num_rows, max_num_cols)


# TODO: make this take a filter_type, perhaps SwiftCometPipeline
def stack_epoch_into_sum_and_median(
    swift_data: SwiftData,
    epoch: Epoch,
    do_coincidence_correction: bool,
    pixel_resolution: SwiftPixelResolution,
) -> Tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage] | None:
    """
    Blindly takes every entry in the given Epoch and attempts to stack it - epoch should be pre-filtered because
    no checks are made here
    If successful, returns a tuple of images: (sum, median, exposure_map)
    The exposure_map image has pixels with values in units of seconds - the total exposure time from the stack of images involved
    """

    obsids = epoch.OBS_ID
    img_filenames = epoch.FITS_FILENAME

    # # getting the image directory is not valid for event mode! those live in uvot/event/
    # img_paths = [
    #     swift_data.get_uvot_image_directory(obsid=x) / y
    #     for x, y in zip(obsids, img_filenames)
    # ]

    # build list of images

    # determine how big our stacked image needs to be
    stacking_image_size = determine_stacking_image_size(
        img_list=img_list, comet_center_coords=comet_center_coords
    )

    if stacking_image_size is None:
        print("Could not determine stacking image size!  Not stacking.")
        return None

    image_data_to_stack: List[SwiftUVOTImage] = []
    exposure_times: List[float] = []
    exposure_map_list = []

    # TODO: we should return the dataframe of images that actually get used to keep track of what gets omitted
    stacking_progress_bar = tqdm(epoch.iterrows(), total=len(epoch), unit="images")
    for _, row in stacking_progress_bar:
        obsid = row["OBS_ID"]

        image_path = swift_data.get_uvot_image_directory(obsid=obsid) / row["FITS_FILENAME"]  # type: ignore

        exp_time = float(row["EXPOSURE"])

        # read the image
        image_data = fits.getdata(image_path, ext=row["EXTENSION"])

        if image_data is None:
            print(f"Could not read fits image at {image_path}! Skipping it in stack.")
            continue

        # do we use the horizons data, or did the user manually tell us where the comet is?
        comet_center_coords = get_comet_center_prefer_user_coords(row=row)

        # check if the comet center is outside the bounds of the image and omit it
        img_height, img_width = image_data.shape  # type: ignore
        if comet_center_coords.x < 0 or comet_center_coords.x > img_width:
            print(f"Image dimensions ==> width={img_width}\theight={img_height}")
            print(f"Invalid comet x coordinate {comet_center_coords.x}! Skipping.")
            continue
        if comet_center_coords.y < 0 or comet_center_coords.y > img_height:
            print(f"Image dimensions ==> width={img_width}\theight={img_height}")
            print(f"Invalid comet y coordinate {comet_center_coords.y}! Skipping.")
            continue

        # new image with the comet nucleus centered
        image_data = center_image_on_coords(
            source_image=image_data,  # type: ignore
            source_coords_to_center=comet_center_coords,
            stacking_image_size=stacking_image_size,
        )

        # do any processing before stacking
        if do_coincidence_correction:
            # TODO: for large event mode images, this is so slow that it is unusable
            # the correction expects images in count rate, but we are storing the raw images so divide by exposure time here
            coi_map = coincidence_correction(
                img=image_data / exp_time, scale=pixel_resolution
            )
            image_data = image_data * coi_map

        image_data_to_stack.append(image_data)
        exposure_times.append(exp_time)

        dead_pixels = image_data == 0
        good_pix = np.ones_like(image_data) * exp_time
        good_pix[dead_pixels] = 0
        exposure_map_list.append(good_pix)

        stacking_progress_bar.set_description(
            f"{image_path.name} extension {row.EXTENSION}"
        )

    if len(image_data_to_stack) == 0:
        print("No valid stacking data left!")
        return None

    final_exposure_map = np.sum(exposure_map_list, axis=0)

    exposure_time = epoch.EXPOSURE.sum()

    # divide by total exposure time so that pixels are count rates
    stack_sum = np.sum(image_data_to_stack, axis=0) / exposure_time

    # divide each image by its exposure time for each image to be in count rate, then take median
    for img, exp_time in zip(image_data_to_stack, exposure_times):
        img /= exp_time
    stack_median = np.median(image_data_to_stack, axis=0)

    return stack_sum, stack_median, final_exposure_map

    # def is_stackable(self, epoch: Epoch) -> bool:
    #     """
    #     Checks that all uw1 and uvv images in this epoch are taken with the same DATAMODE keyword
    #     """
    #
    #     # count the number of unique datamodes: this has to be 1 if we want to stack
    #     return epoch.DATAMODE.nunique() == 1
    #


# TODO: Priority 1: rewrite this
def make_uw1_and_uvv_stacks(
    swift_data: SwiftData,
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    do_coincidence_correction: bool = True,
    remove_vetoed: bool = True,
) -> None:
    """
    Produces sum- and median-stacked images for the uw1 and uvv filters
    The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
    """

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    pre_veto_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    assert pre_veto_epoch is not None

    # filter out the manually vetoed images from the epoch dataframe?
    if remove_vetoed:
        post_veto_epoch = pre_veto_epoch[pre_veto_epoch.manual_veto == np.False_]
    else:
        post_veto_epoch = pre_veto_epoch

    # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    if not is_epoch_stackable(epoch=post_veto_epoch):
        print("Images in the requested stack have mixed data modes! Skipping.")
        return
    else:
        print(
            f"All images taken with FITS keyword DATAMODE={post_veto_epoch.DATAMODE.iloc[0].value}, stacking..."
        )

    # now get just the uw1 and uvv images
    stacked_epoch_mask = np.logical_or(
        post_veto_epoch.FILTER == SwiftFilter.uw1,
        post_veto_epoch.FILTER == SwiftFilter.uvv,
    )
    epoch_to_stack = post_veto_epoch[stacked_epoch_mask]

    # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    if not is_epoch_stackable(epoch=epoch_to_stack):
        print("Images in the requested stack have mixed data modes!")
        num_event_mode_imgs = epoch_to_stack.DATAMODE.value_counts()[
            SwiftImageMode.event_mode
        ]
        num_data_mode_imgs = epoch_to_stack.DATAMODE.value_counts()[
            SwiftImageMode.data_mode
        ]
        print(
            f"Event mode images: {num_event_mode_imgs}\tData mode images: {num_data_mode_imgs}"
        )
        if num_event_mode_imgs > num_data_mode_imgs:
            selected_data_mode = SwiftImageMode.event_mode
        else:
            selected_data_mode = SwiftImageMode.data_mode
        print(f"Filtering images to use only {selected_data_mode.value} ...")
        epoch_to_stack = epoch_to_stack[
            epoch_to_stack.DATAMODE == selected_data_mode
        ].copy()
    else:
        print(
            f"All images taken with FITS keyword DATAMODE={epoch_to_stack.DATAMODE.iloc[0].value}, stacking..."
        )

    # now epoch_to_stack has no vetoed images, and only contains uw1 or uvv images of the same data mode

    epoch_pixel_resolution = epoch_to_stack.ARCSECS_PER_PIXEL.iloc[0]
    stacked_images = StackedUVOTImageSet({})
    exposure_maps = {}

    # do the stacking
    for filter_type in uw1_and_uvv:
        print(f"Stacking for filter {filter_to_file_string(filter_type)} ...")

        # now narrow down the data to just one filter at a time
        filter_mask = epoch_to_stack["FILTER"] == filter_type
        epoch_only_this_filter = epoch_to_stack[filter_mask]

        stack_result = stack_epoch_into_sum_and_median(
            swift_data=swift_data,
            epoch=epoch_only_this_filter,
            do_coincidence_correction=do_coincidence_correction,
            pixel_resolution=epoch_pixel_resolution,
        )
        if stack_result is None:
            ic(
                f"Stacking image for filter {filter_to_file_string(filter_type)} failed!"
            )
            return

        stacked_images[(filter_type, StackingMethod.summation)] = stack_result[0]
        stacked_images[(filter_type, StackingMethod.median)] = stack_result[1]
        exposure_maps[filter_type] = stack_result[2]

    # Adjust the images from each filter to be the same size
    for stacking_method in sum_and_median:
        (uw1_img, uvv_img) = pad_to_match_sizes(
            img_one=stacked_images[(SwiftFilter.uw1, stacking_method)],
            img_two=stacked_images[(SwiftFilter.uvv, stacking_method)],
        )
        stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
        stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

    # Adjust the exposure maps as well so that they stay the same size as the stacked images
    uw1_exp_map, uvv_exp_map = pad_to_match_sizes(
        img_one=exposure_maps[SwiftFilter.uw1],
        img_two=exposure_maps[SwiftFilter.uvv],
    )

    # push all the data into the products for writing later
    epoch_post_stack_prod = scp.get_product(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert epoch_post_stack_prod is not None
    epoch_post_stack_prod.data = epoch_to_stack

    epoch_summary = get_unstacked_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert epoch_summary is not None
    for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
        hdu = epoch_stacked_image_to_fits(
            epoch_summary=epoch_summary,
            img=stacked_images[(filter_type, stacking_method)],
        )
        # hdu = epoch_stacked_image_to_fits(
        #     epoch=epoch_to_stack, img=stacked_images[(filter_type, stacking_method)]
        # )
        img_prod = scp.get_product(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        assert img_prod is not None
        img_prod.data = hdu

    uw1_exp_map_prod = scp.get_product(
        pf=PipelineFilesEnum.exposure_map,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uw1,
    )
    assert uw1_exp_map_prod is not None
    uw1_exp_map_prod.data = epoch_stacked_image_to_fits(
        epoch_summary=epoch_summary, img=uw1_exp_map
    )
    uvv_exp_map_prod = scp.get_product(
        pf=PipelineFilesEnum.exposure_map,
        epoch_id=epoch_id,
        filter_type=SwiftFilter.uvv,
    )
    assert uvv_exp_map_prod is not None
    uvv_exp_map_prod.data = epoch_stacked_image_to_fits(
        epoch_summary=epoch_summary, img=uvv_exp_map
    )


def write_uw1_and_uvv_stacks(scp: SwiftCometPipeline, epoch_id: EpochID) -> None:
    """
    Writes the stacked epoch dataframe, along with the four images created during stacking, and exposure map
    This is a separate step so that the stacking results can be viewed before deciding to save or not save the results
    This assumes that the stacked images are stored in the SwiftCometPipeline object, ready for writing to file
    """
    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    stacked_epoch = scp.get_product(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert stacked_epoch is not None
    stacked_epoch.write()

    for f, s in product(uw1_and_uvv, sum_and_median):
        img_prod = scp.get_product(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        assert img_prod is not None
        img_prod.write()

    for f in uw1_and_uvv:
        em_prod = scp.get_product(
            pf=PipelineFilesEnum.exposure_map, epoch_id=epoch_id, filter_type=f
        )
        assert em_prod is not None
        em_prod.write()


def get_stacked_image_set(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> StackedUVOTImageSet | None:
    stacked_image_set = {}

    uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    sum_and_median = [StackingMethod.summation, StackingMethod.median]

    for f, s in product(uw1_and_uvv, sum_and_median):
        img_data = scp.get_product_data(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=f,
            stacking_method=s,
        )
        if img_data is None:
            return None
        # img_data includes img_data.header for the FITS header, and img_data.data for the numpy image array
        if img_data.data is None:
            return None
        stacked_image_set[f, s] = img_data.data

    return stacked_image_set
