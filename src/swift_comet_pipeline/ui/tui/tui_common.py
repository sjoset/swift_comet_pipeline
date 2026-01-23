from rich.console import Console

from swift_comet_pipeline.builders.build_dispatcher import do_build
from swift_comet_pipeline.pipeline.product_system.dependency_dag import (
    ProductBuildStatus,
    build_toposorter,
    calculate_statuses,
    first_with_build_status,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)


def show_pipeline_status_for_product(
    scp: Products, ref: ProductReference, silent_if_complete: bool = True
) -> None:
    console = Console()
    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)
    for ref, stat in stat_dict.items():
        if stat.build_status == ProductBuildStatus.complete and silent_if_complete:
            continue
        console.print(ref)
        console.print(" -->  ", end="")
        console.print(stat)


# TODO: rewrite this to take bool for looping until the indicated product is built
def build_product_reference(
    scp: Products, ref: ProductReference, verbose: bool = False, force: bool = False
) -> None:

    if verbose:
        print(f"Calculating dependencies for {ref.kind} --> {ref.key}")

    if force:
        print(f"Building {ref.kind} --> {ref.key}")
        do_build(scp=scp, ref=ref)
        return

    ts = build_toposorter(scp=scp, target_product=ref)
    stat_dict = calculate_statuses(scp=scp, ts=ts)

    # show_pipeline_status_for_product(scp=scp, ref=ref)

    first_ready = first_with_build_status(
        stat_dict=stat_dict, status=ProductBuildStatus.ready
    )
    first_regen = first_with_build_status(
        stat_dict=stat_dict, status=ProductBuildStatus.need_regen
    )
    first_stale = first_with_build_status(
        stat_dict=stat_dict, status=ProductBuildStatus.stale
    )

    first_build = None
    first_build = first_ready or first_regen or first_stale
    if first_build is None:
        print("Everything seems to be ready! Skipping build.")
        print("")
        return

    if verbose:
        print(f"Building {first_build.kind} --> {first_build.key}")
    do_build(scp=scp, ref=first_build)
    scp.regenerate()


def build_product_reference_loop(
    scp: Products, ref: ProductReference, verbose: bool = False, force: bool = False
) -> None:

    if verbose:
        print(f"Calculating dependencies for {ref.kind} --> {ref.key}")

    while True:
        ts = build_toposorter(scp=scp, target_product=ref)
        stat_dict = calculate_statuses(scp=scp, ts=ts)

        show_pipeline_status_for_product(scp=scp, ref=ref)

        if stat_dict[ref].build_status == ProductBuildStatus.complete:
            if force:
                do_build(scp=scp, ref=ref)
                break
            else:
                # print(f"Product built!")
                break

        first_ready = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.ready
        )
        first_regen = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.need_regen
        )
        first_stale = first_with_build_status(
            stat_dict=stat_dict, status=ProductBuildStatus.stale
        )

        first_build = None
        first_build = first_ready or first_regen or first_stale
        if first_build is None:
            print("Everything seems to be ready! Skipping build.")
            print("")
            return

        print(f"Ready to build: {first_build}")
        do_build(scp=scp, ref=first_build)

        if ref.kind == ProductKind.epoch_index:
            # we need to do this after epoch_index to build out epoch subpipelines
            scp.regenerate()

        # wait_for_key()


def get_yes_no() -> bool:
    while True:
        raw_selection = input()
        if raw_selection.lower() in ["y", "yes"]:
            return True
        if raw_selection.lower() in ["n", "no"]:
            return False


def wait_for_key(prompt: str = "Press enter to continue") -> None:
    _ = input(prompt)


def clear_screen() -> None:
    console = Console()
    console.clear()
