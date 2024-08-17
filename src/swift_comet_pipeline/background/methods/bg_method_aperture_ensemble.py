import itertools
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from photutils.aperture import (
    CircularAperture,
    ApertureStats,
)
import matplotlib.pyplot as plt
from astropy.visualization import (
    ZScaleInterval,
)
from tqdm import tqdm

from swift_comet_pipeline.background.background_result import BackgroundResult
from swift_comet_pipeline.swift.count_rate import CountRatePerPixel
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.swift.uvot_image import PixelCoord, SwiftUVOTImage
from swift_comet_pipeline.background.background_determination_method import (
    BackgroundDeterminationMethod,
)


ApertureWalk: TypeAlias = list[PixelCoord]


# TODO: document the fields
@dataclass
class WalkingApertureEnsembleConfig:
    initial_radius_pix: int
    walking_step_distance_pix: int
    max_walk_length: int
    radius_growth_step_pix: int
    num_max_radius_growths: int
    growth_dmax_limit_uw1: float
    growth_dmax_limit_uvv: float


def get_walking_aperture_ensemble_config() -> WalkingApertureEnsembleConfig:
    return WalkingApertureEnsembleConfig(
        initial_radius_pix=20,
        walking_step_distance_pix=3,
        max_walk_length=1000,
        radius_growth_step_pix=1,
        num_max_radius_growths=400,
        growth_dmax_limit_uw1=1e-3,
        growth_dmax_limit_uvv=3e-3,
    )


def make_failed_aperture_ensemble_result(err_msg: str) -> BackgroundResult:
    return BackgroundResult(
        count_rate_per_pixel=CountRatePerPixel(value=np.nan, sigma=np.nan),
        params={"error": err_msg},
        method=BackgroundDeterminationMethod.walking_aperture_ensemble,
    )


def scale_image_max_to_one(img: SwiftUVOTImage) -> SwiftUVOTImage:
    return img / np.max(img)


def is_valid_aperture(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    ap: CircularAperture,
    helio_r_au: float,
) -> bool:
    exposure_stats = ApertureStats(exposure_map, ap)

    # We need to relax the exposure threshold at large AU
    # this gives 0.95 at helio_r_au = 1.0 and 0.8 at helio_r_au = 6.0
    exposure_threshold = -0.006 * helio_r_au**2 + 0.012 * helio_r_au + 0.944

    # exposure test: is the average exposure in this aperture at least threshold % of the max exposure time?
    if exposure_stats.mean < exposure_threshold * np.max(exposure_map):
        return False

    # If the min is zero, we are including an area near the edges of the image, which we want to avoid
    # If the standard deviation is zero, then we have no noise from measurements - also suspicious, so reject
    ap_stats = ApertureStats(img, ap)
    if ap_stats.min == 0.0:
        # print("min too low")
        return False

    if ap_stats.std == 0.0:
        # print("std too low")
        return False

    return True


def place_aperture_at_random_valid_position(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    aperture_radius: int,
    helio_r_au: float,
) -> CircularAperture | None:

    img_width = img.shape[1]
    img_height = img.shape[0]

    # exclude entire rows and columns of the image if it contains no exposed pixels - those sections of the image used for padding
    exposure_row = np.sum(exposure_map, axis=1)
    good_rows = exposure_row != 0
    exposure_col = np.sum(exposure_map, axis=0)
    good_cols = exposure_col != 0

    # try a number of times to look for an acceptable spot
    for _ in range(100):
        x = np.random.choice(np.arange(img_width)[good_cols], 1)[0]
        y = np.random.choice(np.arange(img_height)[good_rows], 1)[0]

        test_aperture = CircularAperture((x, y), r=aperture_radius)
        if is_valid_aperture(
            img=img, exposure_map=exposure_map, ap=test_aperture, helio_r_au=helio_r_au
        ):
            return test_aperture

    return None


def do_aperture_walk(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    initial_aperture: CircularAperture,
    helio_r_au: float,
    waec: WalkingApertureEnsembleConfig,
) -> ApertureWalk:
    aperture_radius = initial_aperture.r
    aperture_walk = []

    current_aperture = initial_aperture
    current_aperture_stats = ApertureStats(img, current_aperture)
    aperture_walk.append(
        PixelCoord(x=current_aperture.positions[0], y=current_aperture.positions[1])  # type: ignore
    )

    for _ in range(waec.max_walk_length):
        x = current_aperture.positions[0]  # type: ignore
        y = current_aperture.positions[1]  # type: ignore

        # eight coordinates surrounding (x, y) - left, right, up, down, and diagonals, once we manually removing the center coord (x, y)
        surrounding_coords = list(
            itertools.product(
                [
                    x - waec.walking_step_distance_pix,
                    x,
                    x + waec.walking_step_distance_pix,
                ],
                [
                    y - waec.walking_step_distance_pix,
                    y,
                    y + waec.walking_step_distance_pix,
                ],
            )
        )
        surrounding_coords.remove((x, y))

        # place apertures at these coords
        surrounding_apertures = [
            CircularAperture((coord[0], coord[1]), r=aperture_radius)
            for coord in surrounding_coords
        ]

        # of the eight directions, some apertures may not be valid
        valid_apertures = [
            ap
            for ap in surrounding_apertures
            if is_valid_aperture(
                img=img, exposure_map=exposure_map, ap=ap, helio_r_au=helio_r_au
            )
        ]
        # if none are valid, we're done
        if len(valid_apertures) == 0:
            break

        # look for lowest standard deviation in the new directions
        surrounding_stats = [ApertureStats(img, x) for x in valid_apertures]
        surrounding_stds = [x.std for x in surrounding_stats]
        best_ap = valid_apertures[np.argmin(surrounding_stds)]
        best_ap_stats = surrounding_stats[np.argmin(surrounding_stds)]

        # # TODO: this should never be true because our current_aperture is not in the valid_apertures list!
        # if best_ap == current_aperture:
        #     # all of the directions were worse: stop here
        #     break

        if best_ap_stats.std <= current_aperture_stats.std:
            aperture_walk.append(
                PixelCoord(x=current_aperture.positions[0], y=current_aperture.positions[1])  # type: ignore
            )
            current_aperture = best_ap
            current_aperture_stats = best_ap_stats
        else:
            # the best direction is worse!
            break

    return aperture_walk


def do_aperture_growth(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    initial_aperture: CircularAperture,
    helio_r_au: float,
    waec: WalkingApertureEnsembleConfig,
    filter_type: SwiftFilter,
) -> CircularAperture:
    if filter_type == SwiftFilter.uw1:
        dmax_limit = waec.growth_dmax_limit_uw1
    elif filter_type == SwiftFilter.uvv:
        dmax_limit = waec.growth_dmax_limit_uvv
    else:
        print("Incorrect filter passed! This is a bug!")
        exit(1)

    current_aperture = initial_aperture
    current_aperture_stats = ApertureStats(img, current_aperture)
    for growth_count in range(waec.num_max_radius_growths):
        x = current_aperture.positions[0]  # type: ignore
        y = current_aperture.positions[1]  # type: ignore
        r_pix: float = current_aperture.r  # type: ignore

        grown_ap = CircularAperture((x, y), r=r_pix + waec.radius_growth_step_pix)
        if not is_valid_aperture(
            img=img, exposure_map=exposure_map, ap=grown_ap, helio_r_au=helio_r_au
        ):
            # this new, larger aperture contains under-exposed pixels, so reject it and stop here
            break

        grown_ap_stats = ApertureStats(img, grown_ap)
        if np.abs(current_aperture_stats.max - grown_ap_stats.max) > dmax_limit:
            # this new, larger aperture has a maximum pixel value larger than the smaller one - with a small dmax_limit we can avoid spikes in signal like stars
            break

        current_aperture = grown_ap
        current_aperture_stats = grown_ap_stats

    return current_aperture


def bg_walking_aperture_ensemble(
    img: SwiftUVOTImage,
    exposure_map: SwiftUVOTImage,
    filter_type: SwiftFilter,
    helio_r_au: float,
) -> BackgroundResult:

    waec: WalkingApertureEnsembleConfig = get_walking_aperture_ensemble_config()
    walked_apertures = []
    walked_aperture_stats = []

    for _ in tqdm(range(300)):
        # set down the aperture somewhere valid
        ap = place_aperture_at_random_valid_position(
            img=img,
            exposure_map=exposure_map,
            aperture_radius=waec.initial_radius_pix,
            helio_r_au=helio_r_au,
        )
        if ap is None:
            continue

        # let it walk
        ap_walk = do_aperture_walk(
            img=img,
            exposure_map=exposure_map,
            initial_aperture=ap,
            helio_r_au=helio_r_au,
            waec=waec,
        )

        # Take the coordinates of the end of the walk
        ap_after_walk = CircularAperture(
            (ap_walk[-1].x, ap_walk[-1].y), r=waec.initial_radius_pix
        )

        # Let it grow in radius at fixed position
        ap_after_growth = do_aperture_growth(
            img=img,
            exposure_map=exposure_map,
            initial_aperture=ap_after_walk,
            helio_r_au=helio_r_au,
            waec=waec,
            filter_type=filter_type,
        )

        walked_apertures.append(ap_after_growth)
        walked_aperture_stats.append(ApertureStats(img, ap_after_growth))

    if len(walked_apertures) == 0:
        return make_failed_aperture_ensemble_result(
            err_msg="Automatic backgrounding failed! Is this image under-exposed?"
        )

    # plot_apertures(img=img, aperture_list=walked_apertures)

    # score_results(
    #     filter_type=filter_type,
    #     walked_apertures=walked_apertures,
    #     walked_aperture_stats=walked_aperture_stats,
    # )

    weights = [x.r**2 for x in walked_apertures]  # type: ignore
    ap_medians = [x.median for x in walked_aperture_stats]

    median_mu = np.average(ap_medians, weights=weights)
    median_mu_std = np.sqrt(np.cov(ap_medians, aweights=weights))

    largest_ap_idx = np.argmax([x.r for x in walked_apertures])
    largest_ap_stats = walked_aperture_stats[largest_ap_idx]

    if filter_type == SwiftFilter.uw1:
        return BackgroundResult(
            count_rate_per_pixel=CountRatePerPixel(
                value=float(median_mu), sigma=float(median_mu_std)
            ),
            method=BackgroundDeterminationMethod.walking_aperture_ensemble,
            params={},
        )
    elif filter_type == SwiftFilter.uvv:
        return BackgroundResult(
            count_rate_per_pixel=CountRatePerPixel(
                value=float(largest_ap_stats.median), sigma=float(largest_ap_stats.std)
            ),
            method=BackgroundDeterminationMethod.walking_aperture_ensemble,
            params={},
        )
    else:
        return make_failed_aperture_ensemble_result(
            err_msg="Bad filter type given for analysis!"
        )


# def score_results(
#     filter_type: SwiftFilter,
#     walked_apertures: list[CircularAperture],
#     walked_aperture_stats: list[ApertureStats],
# ):
#
#     df = pd.DataFrame()
#
#     # print(f"Aggregate results:")
#     avg_of_means = np.mean([x.mean for x in walked_aperture_stats])
#     avg_of_medians = np.mean([x.median for x in walked_aperture_stats])
#     # print(f"{avg_of_means=}\t{avg_of_medians=}")
#     # print("")
#
#     df["avg_of_means"] = [avg_of_means]
#     df["avg_of_medians"] = [avg_of_medians]
#
#     weights = [x.r**2 for x in walked_apertures]  # type: ignore
#     avgs = [x.mean for x in walked_aperture_stats]
#     mu = np.average(avgs, weights=weights)
#     std = np.sqrt(np.cov(avgs, aweights=weights))
#     median_mu = np.average([x.median for x in walked_aperture_stats], weights=weights)
#
#     df["weighted_avg_of_means"] = [mu]
#     df["weighted_avg_of_means_sigma"] = [std]
#     df["weighted_avg_of_medians"] = [median_mu]
#
#     # print(f"Weighted average of averages:")
#     # print(f"{float(mu)=}\t{float(std)=}")
#     # print(f"Weighted average of medians:")
#     # print(f"{float(median_mu)=}")
#     # print("")
#
#     manual_bg_result_dict = {
#         SwiftFilter.uw1: 0.0032778,
#         SwiftFilter.uvv: 0.0336888,
#     }  # epoch 002
#     # manual_bg_result_dict = {
#     #     SwiftFilter.uw1: 0.0018453,
#     #     SwiftFilter.uvv: 0.0219918,
#     # }  # epoch 001
#
#     manual_bg_result = manual_bg_result_dict[filter_type]
#
#     largest_ap_idx = np.argmax([x.r for x in walked_apertures])  # type: ignore
#     largest_ap_stats = walked_aperture_stats[largest_ap_idx]
#
#     df["largest_ap_mean"] = [largest_ap_stats.mean]
#     df["largest_ap_median"] = [largest_ap_stats.median]
#
#     df_acc = df / manual_bg_result
#     df_acc = df_acc.drop(columns="weighted_avg_of_means_sigma")
#
#     print("results:")
#     print(df)
#     print("")
#     print("accuracy:")
#     print(df_acc)
#
#     df_acc.to_csv(
#         f"ap_walk_accuracy_epoch_003_{filter_type}.csv",
#         index=False,
#         mode="a",
#         header=False,
#     )


def plot_apertures(img: SwiftUVOTImage, aperture_list: list[CircularAperture]) -> None:
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(img)

    im1 = ax1.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
    for aperture in aperture_list:
        x = aperture.positions[0]  # type: ignore
        y = aperture.positions[1]  # type: ignore
        r = aperture.r
        ax1.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="red", alpha=0.5))  # type: ignore
    fig.colorbar(im1)
    plt.show()


# def background_by_random_apertures(
#     img: SwiftUVOTImage,
#     exposure_mask: SwiftUVOTImage,
#     iterations: int,
#     sigma: int,
#     print_config=None,
# ) -> tuple[float, float]:
#     max_pixel = np.max(img)
#     img = img / max_pixel
#     results_list = []
#     low_r = 2
#     high_r = 50
#     print(
#         f"\nStarting {iterations} iterations of random sized apertures placed on image with a random radius of pixel size {low_r} to {high_r}"
#     )
#     for i in tqdm(range(iterations)):
#         r = random.randrange(low_r, high_r, 2)
#         inital_aperture = intialize_aperture(image=img, radius=r, mask=exposure_mask)
#         test_result = AptPass(
#             walking_path=None,
#             final_apt=inital_aperture,
#             final_stats=ApertureStats(img, inital_aperture),
#             r_grow=None,
#         )
#         results_list.append(test_result)
#     if print_config.print_final_rand_place:
#         final_walk_list = [x.final_apt for x in results_list]
#         plot_apertures(image=img * max_pixel, aperture_list=final_walk_list)
#     clipped_list = clip_results(list_to_clip=results_list, sigma=sigma)
#     mu, std = get_weighted_average_and_std(input_list=clipped_list)
#     return mu * max_pixel, std
