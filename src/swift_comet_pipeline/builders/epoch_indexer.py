from swift_comet_pipeline.builders.vectorial_model_cacher import (
    cache_all_vectorial_models,
)
from swift_comet_pipeline.data_ingestion.epoch_index.build_epoch_index import (
    build_epoch_index,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products


def do_epoch_index(scp: Products) -> None:
    """
    Build our epoch index that contains entries with relevant data we will want to pass around

    Once complete, runs and caches vectorial models for each heliocentric distance in our dataset
    """

    # p_list = find_perihelia(scp=scp)
    # if p_list is None:
    #     return

    epoch_index = build_epoch_index(scp=scp)
    if epoch_index is None:
        return
    scp.save_epoch_index(epoch_index=epoch_index)

    cache_all_vectorial_models(cfg=scp.cfg, epoch_index=epoch_index)
