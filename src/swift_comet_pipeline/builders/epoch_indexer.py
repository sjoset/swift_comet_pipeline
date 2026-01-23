from swift_comet_pipeline.data_ingestion.epoch_index.build_epoch_index import (
    build_epoch_index,
)
from swift_comet_pipeline.data_ingestion.orbit_data.find_perihelion import (
    find_perihelia,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products


def do_epoch_index(scp: Products) -> None:

    p_list = find_perihelia(scp=scp)
    if p_list is None:
        return

    epoch_index = build_epoch_index(scp=scp)
    if epoch_index is None:
        return
    scp.save_epoch_index(epoch_index=epoch_index)
