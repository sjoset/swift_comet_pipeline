import pathlib

from swift_comet_pipeline.swift_data import SwiftData
from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.tui import epoch_menu, get_yes_no
from swift_comet_pipeline.manual_veto import manual_veto


__all__ = ["veto_epoch_step"]


def veto_epoch_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    if pipeline_files.epoch_products is None:
        print("No epoch files found! Exiting.")
        return
    epoch_product = epoch_menu(pipeline_files)
    if epoch_product is None:
        print("Could not select epoch, exiting.")
        return
    epoch_product.load_product()
    epoch_pre_veto = epoch_product.data_product

    epoch_post_veto = manual_veto(
        swift_data=swift_data,
        epoch=epoch_pre_veto,
        epoch_title=epoch_product.product_path.stem,
    )

    print("Save epoch?")
    save_epoch = get_yes_no()
    if not save_epoch:
        return

    epoch_product.data_product = epoch_post_veto
    epoch_product.save_product()
