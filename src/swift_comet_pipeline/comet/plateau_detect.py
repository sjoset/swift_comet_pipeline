import numpy as np

from scipy.ndimage import uniform_filter1d
from itertools import groupby
from dataclasses import dataclass
from typing import List


@dataclass
class Plateau:
    begin_index: int
    end_index: int


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

    # mask = np.abs(ds / smoothed) < threshold
    mask = np.logical_and(np.abs(dds) < threshold, np.abs(ds) < threshold)

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
        Plateau(begin_index=b, end_index=e)
        for (k, rl, b, e) in full_plat
        if rl >= min_length and bool(k) is True
    ]
