import numpy as np
from astropy.io import fits

from swift_comet_pipeline.image_manipulation.image_recenter import (
    center_image_on_coords,
)
from swift_comet_pipeline.image_manipulation.trim_and_relocate import (
    trim_image_and_relocate_pixel_coords,
)
from swift_comet_pipeline.observationlog.build_observation_log import (
    event_mode_header_to_WCS,
)
from swift_comet_pipeline.observationlog.get_comet_position import (
    get_comet_position_at_time,
)
from swift_comet_pipeline.pipeline_utils.time_conversion import (
    uvot_time_to_astropy_time,
)
from swift_comet_pipeline.stacking.determine_stack_size import (
    determine_stacking_image_size,
)
from swift_comet_pipeline.swift.coincidence_correction import coincidence_correction
from swift_comet_pipeline.swift.get_uvot_image_center import get_uvot_image_center
from swift_comet_pipeline.tui.tui_common import wait_for_key
from swift_comet_pipeline.types.pixel_coord import PixelCoord
from swift_comet_pipeline.types.stacking import (
    EventModeTimeBinImageResult,
    StackableUVOTImage,
    StackableUVOTImagePrecursor,
)
from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


def slice_event_mode_image_data(
    event_mode_bintable: fits.BinTableHDU, x_size: int, y_size: int, num_slices: int
) -> tuple[list[SwiftUVOTImage], list[float]]:
    """
    X and Y are indexed from 1, so we subtract one to turn them into 0-indexed coordinates compatible with numpy arrays

    This means we also need to subtract one from the X and Y coordinates that we get for the comet, because the WCS
    is tied to the header information - which assumes 1-indexed coordinates
    """
    ts, xs, ys = (
        event_mode_bintable["TIME"],  # type: ignore
        event_mode_bintable["X"] - 1,  # type: ignore
        event_mode_bintable["Y"] - 1,  # type: ignore
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
    precursor_img: StackableUVOTImagePrecursor,
    num_time_slices: int,
    do_coincidence_correction: bool,
) -> EventModeTimeBinImageResult:
    """
    Header entry TLMAX6 = maximum extent of 'X' values, but not necessarily the largest X that was observed (np.min(['X']) is lower than this value)
    Similar for TLMAX7 for the y value

    Returns image coincidence-corrected, comet-centered set of images in counts (not counts per second)
    """
    ev_hdr: fits.Header = precursor_img.img_hdr
    img_wcs = event_mode_header_to_WCS(hdr=ev_hdr)

    if not isinstance(precursor_img.img, fits.FITS_rec):
        print("Precursor image passed into event mode but the data is not a FITS_rec!")
        print(
            f"{precursor_img.data_mode=} {precursor_img.horizons_id=} {precursor_img.exposure_time_s=} {precursor_img.comet_center=}"
        )
        print(f"{type(precursor_img.img)=}")
        print("Press any key to continue")
        wait_for_key()

    # the data stored in the FITS is read into '.img' - in event mode case, this is a BinTableHDU and not a numpy image array
    ev_table: fits.BinTableHDU = precursor_img.img  # type: ignore
    exposure_time = float(ev_hdr["EXPOSURE"])  # type: ignore
    exposure_time_per_slice = exposure_time / num_time_slices

    # print(f"Slicing event mode table into {num_time_slices} slices ...  ", end="")
    x_min, y_min = np.min(ev_table["X"]), np.min(ev_table["Y"])  # type: ignore
    x_max, y_max = np.max(ev_table["X"]), np.max(ev_table["Y"])  # type: ignore

    img_slices_list, mid_time_list = slice_event_mode_image_data(
        event_mode_bintable=ev_table,
        x_size=ev_hdr["TLMAX6"],  # type: ignore
        y_size=ev_hdr["TLMAX7"],  # type: ignore
        num_slices=num_time_slices,
    )
    # print("Done.")

    # convert times to astropy times
    astropy_mid_times = [
        uvot_time_to_astropy_time(uvot_t=t, event_mode_hdr=ev_hdr)
        for t in mid_time_list
    ]

    # print("Performing Horizons lookups ...  ", end="")
    comet_ras_decs = [
        get_comet_position_at_time(h_id=precursor_img.horizons_id, mt=t)
        for t in astropy_mid_times
    ]
    slice_comet_centers = [
        img_wcs.wcs_world2pix(ra, dec, 1) for ra, dec in comet_ras_decs
    ]
    # When we generated the time-binned image frames, we subtracted 1 from each X and Y - the WCS coordinates from the header
    # are off by one coordinate, so adjust them back down to be zero-indexed
    slice_comet_centers_pix = [
        PixelCoord(x=int(np.round(x - 1)), y=int(np.round(y - 1)))
        for x, y in slice_comet_centers
    ]
    # print("Done. Comet centers during slice:")
    # print(slice_comet_centers_pix)

    # trim the images down: the Xs and Ys that are measured by event mode fall within [[x_min, x_max], [y_min, y_max]]
    new_imgs_and_centers = [
        trim_image_and_relocate_pixel_coords(
            img=i, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pc=pc
        )
        for i, pc in zip(img_slices_list, slice_comet_centers_pix)
    ]

    trimmed_imgs = [x[0] for x in new_imgs_and_centers]
    trimmed_comet_centers = [x[1] for x in new_imgs_and_centers]

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
    # print(f"Event mode stack size: {event_stack_size}")

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
        if do_coincidence_correction:
            print("coincidence correcting ...  ", end="")
            coi_map = coincidence_correction(
                img=image_data / (exposure_time_per_slice),
                scale=SwiftPixelResolution.event_mode,
            )
            image_data = image_data * coi_map
        event_mode_images_to_stack.append(image_data)
        print("Done.")

    # TODO: do the exposure map properly
    # just, whatever idc at this point
    exposure_map = np.ones(event_stack_size) * exposure_time

    event_sum = np.sum(event_mode_images_to_stack, axis=0)

    # divide each image by its exposure time for each image to be in count rate, then take median, then go back to counts
    event_median = np.median(
        [i / exposure_time_per_slice for i in event_mode_images_to_stack], axis=0
    )
    event_median *= exposure_time

    sum_result = StackableUVOTImage(
        img=event_sum,
        comet_center=get_uvot_image_center(img=event_sum),
        exposure_time_s=exposure_time,
        data_mode=SwiftImageMode.event_mode,
    )
    median_result = StackableUVOTImage(
        img=event_median,
        comet_center=get_uvot_image_center(img=event_median),
        exposure_time_s=exposure_time,
        data_mode=SwiftImageMode.event_mode,
    )
    exposure_map_result = StackableUVOTImage(
        img=exposure_map,
        comet_center=get_uvot_image_center(img=exposure_map),
        exposure_time_s=exposure_time,
        data_mode=SwiftImageMode.event_mode,
    )

    return EventModeTimeBinImageResult(
        sum=sum_result, median=median_result, exposure_map=exposure_map_result
    )


# def event_mode_fits_to_time_binned_image(
#     horizons_id: str, fits_path: pathlib.Path, extension_id: int, num_time_slices: int
# ) -> tuple[SwiftUVOTImage, SwiftUVOTImage, SwiftUVOTImage]:
#     """
#     Header entry TLMAX6 = maximum extent of 'X' values, but not necessarily the largest X that was observed (np.min(['X']) is lower than this value)
#     Similar for TLMAX7 for the y value
#     """
#     ev_hdr = fits.getheader(fits_path, extension_id)
#     img_wcs = event_mode_header_to_WCS(hdr=ev_hdr)
#     ev_table = fits.getdata(fits_path, extension_id)
#     exposure_time = float(ev_hdr["EXPOSURE"])
#     exposure_time_per_slice = exposure_time / num_time_slices
#
#     print(f"Slicing event mode table into {num_time_slices} slices ...  ", end="")
#     x_min, y_min = np.min(ev_table["X"]), np.min(ev_table["Y"])  # type: ignore
#     x_max, y_max = np.max(ev_table["X"]), np.max(ev_table["Y"])  # type: ignore
#
#     img_slices_list, mid_time_list = slice_event_mode_image_data(
#         event_mode_bintable=ev_table,
#         x_size=ev_hdr["TLMAX6"],
#         y_size=ev_hdr["TLMAX7"],
#         num_slices=num_time_slices,
#     )
#     print("Done.")
#
#     # convert times to astropy times
#     astropy_mid_times = [
#         uvot_time_to_astropy_time(uvot_t=t, event_mode_hdr=ev_hdr)
#         for t in mid_time_list
#     ]
#
#     print("Performing Horizons lookups ...  ", end="")
#     comet_ras_decs = [
#         get_comet_position_at_time(h_id=horizons_id, mt=t) for t in astropy_mid_times
#     ]
#     slice_comet_centers = [
#         img_wcs.wcs_world2pix(ra, dec, 1) for ra, dec in comet_ras_decs
#     ]
#     slice_comet_centers_pix = [
#         PixelCoord(x=int(np.round(x)), y=int(np.round(y)))
#         for x, y in slice_comet_centers
#     ]
#     print("Done. Comet centers during slice:")
#     print(slice_comet_centers_pix)
#
#     # trim the images down: the Xs and Ys that are measured by event mode fall within [[x_min, x_max], [y_min, y_max]]
#     new_imgs_and_centers = [
#         trim_image_and_relocate_pixel_coords(
#             img=i, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pc=pc
#         )
#         for i, pc in zip(img_slices_list, slice_comet_centers_pix)
#     ]
#
#     trimmed_imgs = [x[0] for x in new_imgs_and_centers]
#     trimmed_comet_centers = [x[1] for x in new_imgs_and_centers]
#     print("Comet centers after trim:")
#     print(trimmed_comet_centers)
#
#     # search_aps = [CircularAperture((pc.x, pc.y), r=10) for pc in trimmed_comet_centers]
#     # peak_finding_comet_centers = [
#     #     find_comet_center(
#     #         img=i, method=CometCenterFindingMethod.aperture_peak, search_aperture=sa
#     #     )
#     #     for i, sa in zip(trimmed_imgs, search_aps)
#     # ]
#
#     # don't bother with peak finding
#     peak_finding_comet_centers = trimmed_comet_centers
#
#     # see how big our final product needs to be
#     event_stack_size = determine_stacking_image_size(
#         img_list=trimmed_imgs, comet_center_coords=peak_finding_comet_centers
#     )
#     assert event_stack_size is not None
#     print(f"Event mode stack size: {event_stack_size}")
#
#     event_mode_images_to_stack = []
#     for i, (trimmed_img, peak_finding_comet_center) in enumerate(
#         zip(trimmed_imgs, peak_finding_comet_centers)
#     ):
#         print(
#             f"Processing image {i+1}/{len(trimmed_imgs)}: centering on {peak_finding_comet_center} ...  ",
#             end="",
#         )
#         image_data = center_image_on_coords(
#             source_image=trimmed_img,
#             source_coords_to_center=peak_finding_comet_center,
#             stacking_image_size=event_stack_size,
#         )
#         print("coincidence correcting ...  ", end="")
#         coi_map = coincidence_correction(
#             img=image_data / (exposure_time_per_slice),
#             scale=SwiftPixelResolution.event_mode,
#         )
#         image_data = image_data * coi_map
#         event_mode_images_to_stack.append(image_data)
#         print("Done.")
#
#     # just, whatever idc at this point
#     exposure_map = np.ones(event_stack_size)
#
#     # divide by total exposure time so that pixels are count rates
#     event_sum = np.sum(event_mode_images_to_stack, axis=0) / exposure_time
#
#     # divide each image by its exposure time for each image to be in count rate, then take median
#     event_median = np.median(
#         [i / exposure_time_per_slice for i in event_mode_images_to_stack], axis=0
#     )
#
#     return event_sum, event_median, exposure_map
