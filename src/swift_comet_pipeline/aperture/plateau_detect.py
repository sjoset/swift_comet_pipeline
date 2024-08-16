from itertools import groupby
from typing import List

from icecream import ic
import numpy as np
from scipy.ndimage import uniform_filter1d

from swift_comet_pipeline.aperture.plateau import Plateau, ProductionPlateau
from swift_comet_pipeline.aperture.q_vs_aperture_radius_entry import (
    QvsApertureRadiusEntry,
)


# TODO: can we use this for a CometRadialProfile as well?
def find_plateaus(
    ys: np.ndarray, xstep: float, smoothing: int, threshold: float, min_length: int
) -> List[Plateau]:
    """
    In this version, ys must be an array sampled at equal distances of xstep between points
    """
    # take average of a window of points: smoothing of 3 takes average of the point and points to the left/right
    smoothed = uniform_filter1d(ys, size=smoothing)

    # compute first and second derivative
    ds = np.gradient(smoothed, xstep)
    dds = np.gradient(ds, xstep)

    # mask = np.abs(ds / smoothed) < threshold
    # mask = np.logical_and(np.abs(dds) < threshold, np.abs(ds) < threshold)
    mask = np.logical_and(
        np.abs(ds * smoothed) < threshold, np.abs(dds * smoothed) < threshold
    )

    # run-length encoding for whether this data point is considered to be locally "flat"
    # consists of the keys k (true/false), and the number of consecutive values of this k
    rle = [(k, sum(1 for _ in g)) for k, g in groupby(mask)]

    # minus 1 to convert to index
    plateau_end_indices = np.cumsum([runlength for _, runlength in rle]) - 1
    plateau_start_indices = np.concatenate(([0], plateau_end_indices[:-1] + 1))

    plateau_indices = list(zip(plateau_start_indices, plateau_end_indices))

    # combine rle and indices
    full_plat = [(k, rl, b, e) for (k, rl), (b, e) in zip(rle, plateau_indices)]

    # construct list of Plateau types for our result
    return [
        Plateau(begin_index=b + 1, end_index=e)
        for (k, rl, b, e) in full_plat
        if rl >= min_length and bool(k) is True and b + 1 < e
    ]

    # squeeze the plateau edges in by one index - this approach catches the 'cliffs' at the right and left edges
    # return [
    #     Plateau(begin_index=b + 1, end_index=e - 1)
    #     for (k, rl, b, e) in full_plat
    #     if rl >= min_length and bool(k) is True and (b + 1) < (e - 1)
    # ]


def find_production_plateaus(
    q_vs_aperture_radius_list: list[QvsApertureRadiusEntry],
) -> list[ProductionPlateau]:
    """
    Make sure the aperture_r_pix entries are all equally spaced!

    Will refuse to look for plateau if there aren't enough positive production values
    """

    rs = np.array([qvsare.aperture_r_pix for qvsare in q_vs_aperture_radius_list])
    r_diffs = np.diff(rs)

    r_step = list(set(r_diffs))[0]
    physical_qs = np.array([qvsare.q_H2O for qvsare in q_vs_aperture_radius_list])

    # ic(rs, r_diffs, r_step, physical_qs)

    positive_production_mask = physical_qs > 0.0
    physical_qs = physical_qs[positive_production_mask]
    if len(physical_qs) < 5:
        return []

    rs = rs[positive_production_mask]

    q_diffs = np.append([1], np.diff(physical_qs))
    qpc = q_diffs / physical_qs

    qs = qpc

    # TODO: magic numbers for the threshold, found empirically - can we do better or justify these?
    thresholds = np.geomspace(start=1e-6, stop=1e-3, num=41, endpoint=True)

    for cur_threshold in thresholds:
        q_plateau_list = find_plateaus(
            ys=qs,
            xstep=r_step,
            # smoothing=np.round(3 / r_step).astype(np.int32),
            smoothing=3,
            threshold=cur_threshold,
            min_length=3 / r_step,
        )
        if len(q_plateau_list) == 0:
            continue

        # print(f"Found plateaus with threshold {cur_threshold:1.3e}...")
        physical_plateaus = [
            ProductionPlateau(
                begin_r=rs[q.begin_index],
                end_r=rs[q.end_index],
                begin_q=physical_qs[q.begin_index],
                end_q=physical_qs[q.end_index],
            )
            for q in q_plateau_list
        ]

        return physical_plateaus

    # print("No plateaus found!")
    return []
