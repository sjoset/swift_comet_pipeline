from swift_comet_pipeline.data_ingestion.observation_log.slice_observation_log_into_epochs import (
    add_epoch_ids_by_time_window,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import Products
from swift_comet_pipeline.ui.mpl_ui.mpl_ui_observation_log_slicing import (
    gui_select_epoch_time_window,
)


def do_epoch_identification(scp: Products) -> None:

    obs_log_df = scp.load_raw_log()
    assert obs_log_df is not None

    dt = gui_select_epoch_time_window(obs_log=obs_log_df)
    df = add_epoch_ids_by_time_window(obs_log=obs_log_df, max_time_between_obs=dt)
    df.attrs = {"time_slice_hours": str(dt.to_value(u.hour))}  # type: ignore

    scp.save_epoch_log(df=df)
