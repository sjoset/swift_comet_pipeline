from operator import itemgetter
from typing import Mapping, TypeAlias
from typing import Any

from swift_comet_pipeline.types.swift_filter import SwiftFilter


Uw1UvvPair: TypeAlias = Mapping[SwiftFilter, Any]
uw1uvv_getter = itemgetter(SwiftFilter.uw1, SwiftFilter.uvv)
