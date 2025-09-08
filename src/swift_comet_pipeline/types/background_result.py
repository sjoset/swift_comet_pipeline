from dataclasses import dataclass
from enum import StrEnum, auto
from types import SimpleNamespace

from swift_comet_pipeline.types.background_determination_method import (
    BackgroundDeterminationMethod,
)


class BackgroundValueEstimator(StrEnum):
    mean = auto()
    median = auto()


@dataclass
class BackgroundResult:
    # estimator of the average background, in *count rate per pixel*
    b_hat: float
    # variance of background pixels
    bg_shot_noise_variance: float
    # number of pixels left in background aperture, after clipping
    bg_num_pixels: float
    # which estimator we used - mean, median
    bg_estimator: BackgroundValueEstimator
    # which method we used to derive background measurement
    method: BackgroundDeterminationMethod
    # any additional information
    params: dict


def background_result_to_dict(
    bg_result: BackgroundResult,
) -> dict:

    # print("To dict:")
    # print(bg_result)
    bg_dict = {
        "b_hat": float(bg_result.b_hat),
        "bg_shot_noise_variance": float(bg_result.bg_shot_noise_variance),
        "bg_num_pixels": float(bg_result.bg_num_pixels),
        "bg_estimator": str(bg_result.bg_estimator),
        "method": str(bg_result.method),
        "params": bg_result.params,
    }

    return bg_dict


# TODO: make result Optional if this can fail somehow
def yaml_dict_to_background_result(raw_yaml: dict) -> BackgroundResult:
    bg = SimpleNamespace(**raw_yaml)
    # print("From yaml:")
    # print(raw_yaml)
    # print("Namespace:")
    # print(bg)
    return BackgroundResult(
        b_hat=float(bg.b_hat),
        bg_shot_noise_variance=float(bg.bg_shot_noise_variance),
        bg_num_pixels=float(bg.bg_num_pixels),
        bg_estimator=BackgroundValueEstimator(bg.bg_estimator),
        method=BackgroundDeterminationMethod(bg.method),
        params=bg.params,
    )


# def background_result_to_dict(
#     bg_result: BackgroundResult,
# ) -> dict:
#     # yaml serializer doesn't support numpy floats for some reason
#     serializable_count_rate = CountRatePerPixel(
#         value=float(bg_result.count_rate_per_pixel.value),
#         sigma=float(bg_result.count_rate_per_pixel.sigma),
#     )
#
#     serializable_bg_result = BackgroundResult(
#         count_rate_per_pixel=serializable_count_rate,
#         bg_aperture_area=bg_result.bg_aperture_area,
#         bg_estimator=bg_result.bg_estimator,
#         params=bg_result.params,
#         method=bg_result.method,
#     )
#     bg_dict = {
#         "params": serializable_bg_result.params,
#         "count_rate_per_pixel": asdict(serializable_bg_result.count_rate_per_pixel),
#         "bg_aperture_area": serializable_bg_result.bg_aperture_area,
#         "bg_estimator": str(serializable_bg_result.bg_estimator),
#         "method": str(serializable_bg_result.method),
#     }
#
#     return bg_dict
#
#
# # TODO: make result Optional if this can fail somehow
# def yaml_dict_to_background_result(raw_yaml: dict) -> BackgroundResult:
#     bg = SimpleNamespace(**raw_yaml)
#     return BackgroundResult(
#         CountRatePerPixel(**bg.count_rate_per_pixel),
#         bg_aperture_area=int(bg.bg_aperture_area),
#         bg_estimator=BackgroundValueEstimator(bg.bg_estimator),
#         params=bg.params,
#         method=BackgroundDeterminationMethod(bg.method),
#     )
