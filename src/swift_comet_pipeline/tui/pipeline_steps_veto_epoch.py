import pathlib

from rich import print as rprint

from swift_comet_pipeline.observationlog.gui_manual_veto import manual_veto

# from swift_comet_pipeline.pipeline.files.data_ingestion_files import DataIngestionFiles
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.tui.tui_common import get_yes_no
from swift_comet_pipeline.tui.tui_menus import epoch_menu


def veto_epoch_step(swift_project_config: SwiftProjectConfig) -> None:
    # data_ingestion_files = DataIngestionFiles(
    #     project_path=swift_project_config.project_path
    # )
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    # if data_ingestion_files.epochs is None:
    #     print("No epoch files found! Exiting.")
    #     return

    epoch_id = epoch_menu(scp=scp)
    if epoch_id is None:
        print("Could not select epoch, exiting.")
        return

    # epoch_product.read()
    # epoch_pre_veto = epoch_product.data
    # if epoch_pre_veto is None:
    #     print(f"Error reading epoch data from {epoch_product.product_path}!")
    #     return
    epoch_pre_veto = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    assert epoch_pre_veto is not None

    epoch_post_veto = manual_veto(
        swift_data=swift_data,
        epoch=epoch_pre_veto,
        epoch_title=epoch_id,
    )

    print("Save changes?")
    save_epoch = get_yes_no()
    if not save_epoch:
        return

    # epoch_id.data = epoch_post_veto
    # epoch_id.write()
    epoch_post_veto_prod = scp.get_product(
        pf=PipelineFilesEnum.epoch_pre_stack, epoch_id=epoch_id
    )
    assert epoch_post_veto_prod is not None
    rprint(
        f"[green]Writing epoch file to {epoch_post_veto_prod.product_path}...[/green]"
    )
    epoch_post_veto_prod.data = epoch_post_veto
    epoch_post_veto_prod.write()
