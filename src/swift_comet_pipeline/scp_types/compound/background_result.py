from dataclasses import asdict, dataclass
from enum import StrEnum, auto
from types import SimpleNamespace

from swift_comet_pipeline.scp_types.primitive.background_determination_method import (
    BackgroundDeterminationMethod,
)


class BackgroundValueEstimator(StrEnum):
    mean = auto()
    median = auto()


@dataclass
class BackgroundResult:
    # estimator of the average background, in *count rate per pixel*, in units of per second per pixel
    b_hat: float
    # variance of background pixels - per pixel**2 per sec**2
    bg_shot_noise_variance: float
    # number of pixels left in background aperture, after clipping
    bg_num_pixels: float
    # which estimator we used - mean, median
    bg_estimator: BackgroundValueEstimator
    # which method we used to derive background measurement
    method: BackgroundDeterminationMethod
    # any additional information
    params: dict


# TODO: transfer this cattrs structure/unstructure
def json_from_background_result(bgr: BackgroundResult) -> dict:
    """
    the 'params' dict should be pre-serialized to strings for json
    """
    return asdict(bgr)


def background_result_from_json(json_dict: dict) -> BackgroundResult:
    bg = SimpleNamespace(**json_dict)
    return BackgroundResult(
        b_hat=float(bg.b_hat),
        bg_shot_noise_variance=float(bg.bg_shot_noise_variance),
        bg_num_pixels=float(bg.bg_num_pixels),
        bg_estimator=BackgroundValueEstimator(bg.bg_estimator),
        method=BackgroundDeterminationMethod(bg.method),
        params=bg.params,
    )
