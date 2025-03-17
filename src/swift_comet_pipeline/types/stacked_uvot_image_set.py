from typing import TypeAlias

from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.swift_uvot_image import SwiftUVOTImage


StackedUVOTImageSet: TypeAlias = dict[
    tuple[SwiftFilter, StackingMethod], SwiftUVOTImage
]
