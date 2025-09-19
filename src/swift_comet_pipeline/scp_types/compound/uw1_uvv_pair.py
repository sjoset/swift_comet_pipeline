from operator import itemgetter
from typing import Mapping, TypeAlias
from typing import Any

from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter


Uw1UvvPair: TypeAlias = Mapping[UvotFilter, Any]
uw1uvv_getter = itemgetter(UvotFilter.uw1, UvotFilter.uvv)
