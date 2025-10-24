from functools import partial
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from graphlib import TopologicalSorter

import numpy as np
import astropy.units as u
from rich.console import RenderResult
from rich.text import Text

from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ProductReference,
    Products,
)


def build_toposorter(
    scp: Products, target_product: ProductReference
) -> TopologicalSorter:
    ts = TopologicalSorter()
    visited: set[ProductReference] = set()

    # depth-first search
    def _dfs(node: ProductReference) -> None:
        if node in visited:
            return
        visited.add(node)
        deps = scp.reg.deps_for(node) or []
        for d in deps:
            _dfs(d)
        ts.add(node, *deps)  # node depends on deps

    _dfs(target_product)
    return ts


class ProductBuildStatus(StrEnum):
    missing = auto()

    # exists but parents are missing
    need_regen = "needs regen"

    # missing, deps complete
    ready = "ready to build"

    # exists, older than a dep
    stale = auto()

    complete = auto()


_STATUS_STYLE = {
    ProductBuildStatus.missing: "red",
    ProductBuildStatus.need_regen: "yellow",
    ProductBuildStatus.ready: "deep_sky_blue1",
    ProductBuildStatus.stale: "orange3",
    ProductBuildStatus.complete: "green bold",
}


@dataclass(frozen=True)
class ProductStatus:
    build_status: ProductBuildStatus
    exists: bool
    mtime: float

    def __str__(self):
        exist_str = "[file exists]" if self.exists else "[no file]"
        date_str = (
            datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M:%S")
            if self.mtime
            else "no file timestamp"
        )
        build_str = f"[{self.build_status}]"
        return f"Status: {exist_str:<15}{build_str:<20}{date_str:<30}"

    def __rich_console__(self, *_) -> RenderResult:
        t = Text("Status: ")

        exist_str = "[file exists]" if self.exists else "[no file]"
        exist_str = f"{exist_str:<15}"
        exist_style = "green" if self.exists else "red"
        t.append(exist_str, style=exist_style)

        build_str = f"[{self.build_status}]"
        build_str = f"{build_str:<20}"
        build_style = _STATUS_STYLE.get(self.build_status, "white bold")
        t.append(build_str, style=build_style)

        date_str = (
            datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M:%S")
            if self.mtime
            else "no file timestamp"
        )
        date_style = "cyan" if self.mtime else "dim"
        t.append(date_str, style=date_style)

        yield t


def safe_mtime(scp: Products, ref: ProductReference) -> float | None:
    p = scp.path_for(ref=ref)
    if p is None:
        return None
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, PermissionError, OSError):
        return None


def calculate_statuses(
    scp: Products, ts: TopologicalSorter
) -> dict[ProductReference, ProductStatus]:

    ref_dep_list = list(ts.static_order())

    existences = [scp.exists(ref=x) for x in ref_dep_list]

    mtimes = [safe_mtime(scp, x) for x in ref_dep_list]

    stale_products = False

    # TODO: test this value and adjust
    # a product can be newer than a parent by amount of seconds below - this allows the subpipelines to be run within this timeframe and be considered consistent
    # because of the inconsistent way that the dependency graph is built - it can change the order of products from subpipelines, which causes false staleness
    # this happens for water production calculations because they pull from a dust and oh subpipeline
    mtime_tolerance_threshold_s = (24 * u.hour).to_value(u.s)

    statuses = {}
    # fill in the status of each product, from first to be built in dep tree to last
    for i, (ref, ref_exists, mtime) in enumerate(zip(ref_dep_list, existences, mtimes)):

        # we have existence and mtime for ProductStatus(), so fill these in and decide build_status with logic below
        status_factory = partial(ProductStatus, exists=ref_exists, mtime=mtime)
        # list of build statuses of parents
        build_statuses = [x.build_status for x in statuses.values()] if statuses else []

        # does this product not exist?
        if not ref_exists:

            # then we are ready if all parents are complete
            if all([ProductBuildStatus.complete == x for x in build_statuses]):
                # print(
                #     f"Marking {ref} as ready because it is missing and all parents complete!"
                # )
                statuses[ref] = status_factory(build_status=ProductBuildStatus.ready)

                # stale_products = True
                # print(f"Marking children of {ref} as stale!")
                continue

            # not all parents were complete, so we are just missing
            # print(f"Marking {ref} as missing!")
            statuses[ref] = status_factory(build_status=ProductBuildStatus.missing)
            continue

        # it exists - is there any missing parent?
        if ProductBuildStatus.missing in build_statuses:
            # print(f"Marking {ref} as needing regeneration because parent is missing!")
            statuses[ref] = status_factory(build_status=ProductBuildStatus.need_regen)
            continue

        # it exists - is there any parent that is ready to build? Then we need rebuilding after it
        if ProductBuildStatus.ready in build_statuses:
            # print(
            #     f"Marking {ref} as needing regeneration because it exists but parent is ready to build!"
            # )
            statuses[ref] = status_factory(build_status=ProductBuildStatus.need_regen)
            continue

        # TODO: staleness problem
        # TODO: when we generate dep tree using two epoch subpipelines, the same kind of product gets grouped next to each other -
        # which means our dep tree will insist we do both backgrounds one after the other, then subtract one after the other, etc.
        # If instead we generate products in the subpipelines separately, one subpipeline will be older and marked stale because the
        # other subpipeline's results are newer
        # Either we:
        # 1) standardize the dependency order,
        # 2) we only generate dag trees using the final product but stop building when we hit our target
        # 3) *only* allow building the final products, so that only one dep tree is generated and can't conflict with anything else
        # 4) when marking as stale, check if the product is the same subpipeline - if not, we don't care?
        # 5) change the mtime comparison to allow for ~5 minutes, 1 day, etc threshold before it's considered stale
        # Decided on number 5 but this seems suboptimal

        # it exists - is there any parent that is newer?
        assert mtime is not None
        prev_mtimes = mtimes[:i]
        if any(
            [
                mtime + mtime_tolerance_threshold_s < x if x else None
                for x in prev_mtimes
            ]
        ):
            print(
                f"Marking {ref.kind.value, ref.key} as stale because a parent is newer!"
            )
            # print(f"Previous mtimes: {prev_mtimes} {np.array(prev_mtimes) - mtime}")
            # print(
            #     f"{ [(x.kind.value, m - mtime) for x, m in list(mtime_dict.items())[:i]] }"
            # )
            # report = [
            #     (x.kind.value, x.key, m - mtime)
            #     for x, m in list(mtime_dict.items())[:i]
            # ]
            # print(report)
            statuses[ref] = status_factory(build_status=ProductBuildStatus.stale)
            stale_products = True
            continue

        # all parents exist - but are we newer than the guys that come after us?
        # the rest are stale, but we are complete
        assert mtime is not None
        remaining_mtimes = mtimes[i + 1 :]
        if any(
            [
                mtime - mtime_tolerance_threshold_s > x if x else None
                for x in remaining_mtimes
            ]
        ):
            print(f"Marking {ref} as complete and the rest as stale!")
            statuses[ref] = status_factory(build_status=ProductBuildStatus.complete)
            # TODO: staleness problem
            stale_products = True
            continue

        # ok - are any parents stale? then we are too
        if stale_products:
            # print(f"Marking {ref} as stale because of stale parent!")
            statuses[ref] = status_factory(build_status=ProductBuildStatus.stale)
            continue

        # does any parent need regen? then we do too
        if ProductBuildStatus.need_regen in build_statuses:
            # print(f"Marking {ref} as needing regeneration because a parent does!")
            statuses[ref] = status_factory(build_status=ProductBuildStatus.need_regen)
            continue

        # if we got here, then we must be complete
        statuses[ref] = status_factory(build_status=ProductBuildStatus.complete)

    return statuses
