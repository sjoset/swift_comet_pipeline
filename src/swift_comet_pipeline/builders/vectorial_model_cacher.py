from functools import partial

from joblib import Parallel, delayed
import astropy.units as u

from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    vectorial_model_settings_init,
    water_vectorial_model,
)
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex


def get_vectorial_model_cache_Q() -> u.Quantity:
    return 1e28 / u.s


def _parallel_vectorial_model_runner(cfg: CometProjectConfig, rh_au: float) -> None:
    base_q = get_vectorial_model_cache_Q()
    vectorial_model_settings_init(comet_project_config=cfg)
    _ = water_vectorial_model(base_q=base_q, helio_r=rh_au * u.AU)


def cache_all_vectorial_models(
    cfg: CometProjectConfig, epoch_index: EpochIndex, n_jobs: int = -1
) -> None:

    print(f"Caching vectorial models...")

    rh_au_list = list(set([eid.rh_au for eid in epoch_index]))

    print(f"Found {len(rh_au_list)} heliocentric distances:")
    for rh_au in rh_au_list:
        print(f"{rh_au:2.4f} AU")

    builder_func = partial(_parallel_vectorial_model_runner, cfg=cfg)
    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(builder_func)(rh_au=rh_au) for rh_au in rh_au_list
    )
