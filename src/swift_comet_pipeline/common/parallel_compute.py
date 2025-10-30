from collections.abc import Callable

import numpy as np
from joblib import delayed, Parallel, cpu_count
from tqdm import tqdm


def parallel_compute_float(
    f: Callable, xs: np.ndarray, do_tqdm: bool = False
) -> np.ndarray:

    iterator = (delayed(f)(x) for x in xs)
    if do_tqdm:
        iterator = tqdm(iterator, total=len(xs), desc="Computing active area...")

    out = Parallel(
        n_jobs=cpu_count(),
        backend="multiprocessing",
        batch_size="auto",
        prefer=None,
        verbose=0,
    )(iterator)

    return np.asarray(out, dtype=float)
