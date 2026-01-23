from swift_comet_pipeline.data_ingestion.veto.gui_manual_veto import manual_veto
from swift_comet_pipeline.pipeline.product_system.product_facade import Products


def do_image_veto(scp: Products) -> None:

    print("Starting veto..")
    epoch_log_df = scp.load_epoch_log()
    assert epoch_log_df is not None

    # TODO: get results and do the writing here instead of in manual_veto
    manual_veto(scp=scp)
