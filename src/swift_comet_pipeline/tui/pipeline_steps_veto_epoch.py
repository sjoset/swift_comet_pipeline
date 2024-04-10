import pathlib
from rich import print as rprint

from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.projects.configs import SwiftProjectConfig

from swift_comet_pipeline.tui.tui_common import epoch_menu, get_yes_no, wait_for_key
from swift_comet_pipeline.pipeline.manual_veto import manual_veto
from swift_comet_pipeline.pipeline.pipeline_files import (
    PipelineFiles,
    PipelineProductType,
)


def veto_epoch_step(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    swift_data = SwiftData(data_path=pathlib.Path(swift_project_config.swift_data_path))

    num_epoch_ids = pipeline_files.get_epoch_ids()
    if num_epoch_ids is None:
        print("No epoch files found! Exiting.")
        wait_for_key()
        return

    epoch_id = epoch_menu(pipeline_files)
    if epoch_id is None:
        print("Could not select epoch, exiting.")
        wait_for_key()
        return

    epoch_pre_veto = pipeline_files.read_pipeline_product(
        PipelineProductType.epoch, epoch_id=epoch_id
    )
    if epoch_pre_veto is None:
        print("Error reading epoch!")
        wait_for_key()
        return

    epoch_post_veto = manual_veto(
        swift_data=swift_data,
        epoch=epoch_pre_veto,
        epoch_title=epoch_id,
    )

    print("Save changes?")
    save_epoch = get_yes_no()
    if not save_epoch:
        return

    pipeline_files.write_pipeline_product(
        PipelineProductType.epoch, data=epoch_post_veto, epoch_id=epoch_id
    )
    rprint("[green]Epoch saved.[/green]")
    wait_for_key()
