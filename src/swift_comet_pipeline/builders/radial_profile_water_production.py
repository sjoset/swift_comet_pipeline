from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductReference,
    Products,
    WaterProductionKey,
)


def do_radial_profile_water_production(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, WaterProductionKey)

    # load the epoch info
    eid = scp.load_epoch_index_entry(epoch_id=ref.key.epoch_id)

    oh_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.oh_filter,
        stacking_method=ref.key.stacking_method,
    )
    dust_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
    )

    # load the radial profiles from the oh and dust filters
    oh_rad_prof = scp.load_extracted_radial_profile(key=oh_key)
    dust_rad_prof = scp.load_extracted_radial_profile(key=dust_key)

    # do continuum subtraction to isolate oh column density
    # oh_col_dens = calculate_col

    # fit column density to vectorial model

    # package results and write

    return
