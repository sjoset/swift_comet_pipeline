import numpy as np

from scipy.ndimage import uniform_filter1d
from itertools import groupby
from dataclasses import dataclass
from typing import List

from swift_comet_pipeline.water_production.q_vs_aperture_radius import (
    QvsApertureRadiusEntry,
)


@dataclass
class Plateau:
    begin_index: int
    end_index: int


@dataclass
class PhysicalPlateau:
    begin_r: float
    end_r: float
    begin_q: float
    end_q: float


# TODO: can we use this for a CometRadialProfile as well?
def plateau_detect(
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

    # print(f"{ds=}")
    # print(f"{dds=}")

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
    q_vs_aperture_list: list[QvsApertureRadiusEntry],
) -> list[PhysicalPlateau] | None:
    """
    Make sure the aperture_r_pix entries are all equally spaced!
    """

    rs = np.array([qvsare.aperture_r_pix for qvsare in q_vs_aperture_list])
    r_diffs = np.diff(rs)

    r_step = list(set(r_diffs))[0]
    physical_qs = np.array([qvsare.q_H2O for qvsare in q_vs_aperture_list])

    positive_production_mask = physical_qs > 0.0
    physical_qs = physical_qs[positive_production_mask]
    if len(physical_qs) < 5:
        return None

    rs = rs[positive_production_mask]

    # try to scale the qs to something reasonably predictable
    # qs = physical_qs / np.max(physical_qs)
    # print(f"{qs=} {qs.shape=}")

    q_diffs = np.append([1], np.diff(physical_qs))
    # print(f"{q_diffs=} {q_diffs.shape=}")
    qpc = q_diffs / physical_qs
    # print(f"{qpc=} {qpc.shape=}")

    qs = qpc

    print(f"Plateau search at dust redness {q_vs_aperture_list[0].dust_redness}:")

    # thresholds = np.geomspace(start=1e-1, stop=10, num=41, endpoint=True) / r_step
    thresholds = np.geomspace(start=1e-6, stop=1e-3, num=41, endpoint=True)
    # thresholds = np.linspace(start=0.001, stop=0.05, num=31, endpoint=True)

    for cur_threshold in thresholds:
        # print(f"{cur_threshold=}")
        q_plateau_list = plateau_detect(
            ys=qs,
            xstep=r_step,
            smoothing=3,
            threshold=cur_threshold,
            min_length=3 / r_step,
        )
        if len(q_plateau_list) == 0:
            continue

        print(f"Found plateaus with threshold {cur_threshold:1.3e}...")
        physical_plateaus = [
            PhysicalPlateau(
                begin_r=rs[q.begin_index],
                end_r=rs[q.end_index],
                begin_q=physical_qs[q.begin_index],
                end_q=physical_qs[q.end_index],
            )
            for q in q_plateau_list
        ]

        return physical_plateaus

    print("No plateaus found!")
    return None
