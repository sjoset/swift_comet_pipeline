from dataclasses import asdict, dataclass
from enum import StrEnum, auto
from types import SimpleNamespace

from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)
from swift_comet_pipeline.types.count_rate import CountRatePerPixel


class BackgroundValueEstimator(StrEnum):
    mean = auto()
    median = auto()


@dataclass
class BackgroundResult:
    count_rate_per_pixel: CountRatePerPixel
    bg_aperture_area: float
    bg_estimator: BackgroundValueEstimator
    method: BackgroundDeterminationMethod
    params: dict


def background_result_to_dict(
    bg_result: BackgroundResult,
) -> dict:
    # yaml serializer doesn't support numpy floats for some reason
    serializable_count_rate = CountRatePerPixel(
        value=float(bg_result.count_rate_per_pixel.value),
        sigma=float(bg_result.count_rate_per_pixel.sigma),
    )

    serializable_bg_result = BackgroundResult(
        count_rate_per_pixel=serializable_count_rate,
        bg_aperture_area=bg_result.bg_aperture_area,
        bg_estimator=bg_result.bg_estimator,
        params=bg_result.params,
        method=bg_result.method,
    )
    bg_dict = {
        "params": serializable_bg_result.params,
        "count_rate_per_pixel": asdict(serializable_bg_result.count_rate_per_pixel),
        "bg_aperture_area": serializable_bg_result.bg_aperture_area,
        "bg_estimator": str(serializable_bg_result.bg_estimator),
        "method": str(serializable_bg_result.method),
    }

    return bg_dict


# TODO: make result Optional if this can fail somehow
def yaml_dict_to_background_result(raw_yaml: dict) -> BackgroundResult:
    bg = SimpleNamespace(**raw_yaml)
    return BackgroundResult(
        CountRatePerPixel(**bg.count_rate_per_pixel),
        bg_aperture_area=int(bg.bg_aperture_area),
        bg_estimator=BackgroundValueEstimator(bg.bg_estimator),
        params=bg.params,
        method=BackgroundDeterminationMethod(bg.method),
    )
