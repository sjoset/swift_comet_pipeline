import numpy as np
from rich import print as rprint
from rich.panel import Panel

from swift_comet_pipeline.aperture.q_vs_aperture_radius import (
    q_vs_aperture_radius_at_epoch,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.tui.tui_common import get_selection
from swift_comet_pipeline.tui.tui_menus import subpipeline_selection_menu


def menu_analyze_all_or_selection() -> str:
    user_selection = None

    print("Analyze all (a), make a selection (s), or quit? (q)")
    while user_selection is None:
        raw_selection = input()
        if raw_selection == "a" or raw_selection == "s" or raw_selection == "q":
            user_selection = raw_selection

    return user_selection


# TODO: show stacked images with aperture radii shaded in given the plateaus it finds
def aperture_analysis_step(swift_project_config: SwiftProjectConfig) -> None:
    scp = SwiftCometPipeline(swift_project_config=swift_project_config)

    # # TODO: select stacking method
    # stacking_method = StackingMethod.summation

    epoch_id_list = scp.get_epoch_id_list()
    if epoch_id_list is None:
        return

    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    selection = get_selection(stacking_methods)
    if selection is None:
        return
    stacking_method = stacking_methods[selection]
    print(f"Stacking method selected: {stacking_method}")

    dust_rednesses = [
        DustReddeningPercent(x)
        for x in np.linspace(start=-100.0, stop=100.0, num=201, endpoint=True)
    ]

    menu_selection = menu_analyze_all_or_selection()
    if menu_selection == "q":
        return

    # for epoch_id in epoch_id_list:
    #     q_vs_aperture_radius_at_epoch(
    #         scp=scp,
    #         epoch_id=epoch_id,
    #         dust_rednesses=dust_rednesses,
    #         stacking_method=stacking_method,
    #     )
    #     print("")

    epoch_ids_to_analyze = None

    if menu_selection == "a":
        epoch_ids_to_analyze = epoch_id_list
    elif menu_selection == "s":
        parent_epoch_id_selection = subpipeline_selection_menu(
            scp=scp, status_marker=SwiftCometPipelineStepEnum.aperture_analysis
        )
        if parent_epoch_id_selection is None:
            return
        epoch_ids_to_analyze = [parent_epoch_id_selection]

    assert epoch_ids_to_analyze is not None

    for epoch_id in epoch_ids_to_analyze:
        rprint(Panel(f"Epoch {epoch_id}:", expand=False))
        q_vs_aperture_radius_at_epoch(
            scp=scp,
            epoch_id=epoch_id,
            dust_rednesses=dust_rednesses,
            stacking_method=stacking_method,
        )
        print("")
