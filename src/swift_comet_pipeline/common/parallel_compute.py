from collections.abc import Callable

import numpy as np
from joblib import delayed, Parallel, cpu_count
from tqdm import tqdm

from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.ui.tui.tui_common import build_product_reference_loop


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


def parallel_loop_builder(
    scp: Products, ref: ProductReference, force: bool = False
) -> None:
    vectorial_model_settings_init(comet_project_config=scp.cfg)
    build_product_reference_loop(scp=scp, ref=ref, force=force)
