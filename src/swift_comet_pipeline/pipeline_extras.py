from enum import StrEnum
from rich import print as rprint

from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline_extras_epoch_summary import (
    pipeline_extra_epoch_summary,
    pipeline_extra_latex_table_summary,
)
from swift_comet_pipeline.pipeline_extras_orbital_data import (
    pipeline_extra_orbital_data,
)
from swift_comet_pipeline.pipeline_extras_status import pipeline_extra_status
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.tui import clear_screen, get_selection, wait_for_key
from swift_comet_pipeline.pipeline_files import PipelineFiles, PipelineProductType


class PipelineExtrasMenuEntry(StrEnum):
    pipeline_status = "pipeline status"
    epoch_summary = "epoch summary"
    epoch_latex_observation_log = "observation summary in latex format"
    testing = "test new pipeline code"
    get_orbital_data = "query jpl for comet and earth orbital data"

    @classmethod
    def all_extras(cls):
        return [x for x in cls]


def pipeline_extras_menu(swift_project_config: SwiftProjectConfig) -> None:
    exit_menu = False
    extras_menu_entries = PipelineExtrasMenuEntry.all_extras()
    while not exit_menu:
        clear_screen()
        step_selection = get_selection(extras_menu_entries)
        if step_selection is None:
            exit_menu = True
            continue
        step = extras_menu_entries[step_selection]

        if step == PipelineExtrasMenuEntry.pipeline_status:
            pipeline_extra_status(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.epoch_summary:
            pipeline_extra_epoch_summary(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.epoch_latex_observation_log:
            pipeline_extra_latex_table_summary(
                swift_project_config=swift_project_config
            )
        elif step == PipelineExtrasMenuEntry.testing:
            do_test(swift_project_config=swift_project_config)
        elif step == PipelineExtrasMenuEntry.get_orbital_data:
            pipeline_extra_orbital_data(swift_project_config=swift_project_config)
        else:
            exit_menu = True


def do_test(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(
        base_product_save_path=swift_project_config.product_save_path
    )

    # cod = pipeline_files.read_pipeline_product(p=PipelineProductType.comet_orbital_data)
    # print(cod)
    # eod = pipeline_files.read_pipeline_product(p=PipelineProductType.earth_orbital_data)
    # print(eod)
    # df = pipeline_files.read_pipeline_product(p=PipelineProductType.observation_log)
    # print(df)

    eids = pipeline_files.get_epoch_ids()
    if eids is None:
        print("No epoch ids!")
        wait_for_key()
        return

    rprint("[red]epoch ids[/red]")
    print(eids)

    rprint("[red]epoch[/red]")
    epoch = pipeline_files.read_pipeline_product(
        p=PipelineProductType.epoch, epoch_id=eids[0]
    )
    print(epoch)
    for eid in eids:
        print(eid, pipeline_files.exists(p=PipelineProductType.epoch, epoch_id=eid))

    rprint("[red]stacked epoch[/red]")
    stacked_epoch = pipeline_files.read_pipeline_product(
        p=PipelineProductType.stacked_epoch, epoch_id=eids[0]
    )
    print(stacked_epoch)
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(p=PipelineProductType.stacked_epoch, epoch_id=eid),
        )

    rprint("[red]stacked image[/red]")
    img = pipeline_files.read_pipeline_product(
        p=PipelineProductType.stacked_image,
        epoch_id=eids[0],
        filter_type=SwiftFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(
                p=PipelineProductType.stacked_image,
                epoch_id=eid,
                filter_type=SwiftFilter.uw1,
                stacking_method=StackingMethod.summation,
            ),
        )

    rprint("[red]img header[/red]")
    hdr = pipeline_files.read_pipeline_product(
        p=PipelineProductType.stacked_image_header,
        epoch_id=eids[0],
        filter_type=SwiftFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    print(hdr)

    rprint("[red]bg analysis[/red]")
    bg = pipeline_files.read_pipeline_product(
        p=PipelineProductType.background_analysis,
        epoch_id=eids[0],
        filter_type=SwiftFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    print(bg)
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(
                p=PipelineProductType.background_analysis,
                epoch_id=eid,
                filter_type=SwiftFilter.uw1,
                stacking_method=StackingMethod.summation,
            ),
        )

    rprint("[red]bg subtracted[/red]")
    bg_img = pipeline_files.read_pipeline_product(
        p=PipelineProductType.background_subtracted_image,
        epoch_id=eids[0],
        filter_type=SwiftFilter.uw1,
        stacking_method=StackingMethod.summation,
    )
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(
                p=PipelineProductType.background_subtracted_image,
                epoch_id=eid,
                filter_type=SwiftFilter.uw1,
                stacking_method=StackingMethod.summation,
            ),
        )

    rprint("[red]q vs ap[/red]")
    q_vs_ap = pipeline_files.read_pipeline_product(
        p=PipelineProductType.qh2o_vs_aperture_radius,
        epoch_id=eids[0],
        stacking_method=StackingMethod.summation,
    )
    print(q_vs_ap)
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(
                p=PipelineProductType.qh2o_vs_aperture_radius,
                epoch_id=eid,
                stacking_method=StackingMethod.summation,
            ),
        )

    rprint("[red]q from profile[/red]")
    qfp = pipeline_files.read_pipeline_product(
        p=PipelineProductType.qh2o_from_profile,
        epoch_id=eids[0],
        stacking_method=StackingMethod.summation,
    )
    print(qfp)
    for eid in eids:
        print(
            eid,
            pipeline_files.exists(
                p=PipelineProductType.qh2o_from_profile,
                epoch_id=eid,
                stacking_method=StackingMethod.summation,
            ),
        )

    wait_for_key()
