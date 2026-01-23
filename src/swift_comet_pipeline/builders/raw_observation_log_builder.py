from swift_comet_pipeline.data_ingestion.observation_log.build_observation_log import (
    build_observation_log,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.swift.swift_data import SwiftData


def do_observation_log_raw(scp: Products) -> None:

    swift_data = SwiftData(data_path=scp.cfg.swift_data_path)
    obs_log_df = build_observation_log(
        swift_data=swift_data, horizons_id=scp.cfg.jpl_horizons_id
    )

    # TODO: we should probably exit here instead
    if obs_log_df is None:
        print("Error while building observation log!  Not saving.")
        return

    scp.save_raw_log(df=obs_log_df)
