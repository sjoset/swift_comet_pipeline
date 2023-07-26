# #!/usr/bin/env python3
#
# import pathlib
# import json
# import itertools
#
# from typing import Dict
#
# # from dataclasses import dataclass
#
# from swift_types import (
#     SwiftFilter,
#     filter_to_string,
#     StackingMethod,
# )
#
# from stacking import read_stacked_image


# __version__ = "0.0.1"
# __all__ = ["stackinfo_from_stacked_images", "stacked_images_from_stackinfo"]


# @dataclass
# class SwiftStackInfo:
#     uw1_median_image_path: str
#     uw1_median_info_path: str
#     uw1_sum_image_path: str
#     uw1_sum_info_path: str
#     uvv_median_image_path: str
#     uvv_median_info_path: str
#     uvv_sum_image_path: str
#     uvv_sum_info_path: str


# # TODO: make the stacking_outputs dict a dataclass of its own?
# def stackinfo_from_stacked_images(
#     stackinfo_output_path: pathlib.Path, stacking_outputs: Dict
# ) -> None:
#     filter_types = [SwiftFilter.uw1, SwiftFilter.uvv]
#     stacking_methods = [StackingMethod.summation, StackingMethod.median]
#
#     # build a record of all files created in this stacking run for use later
#     stack_dict = {}
#     for filter_type, stacking_method in itertools.product(
#         filter_types, stacking_methods
#     ):
#         img_key = f"{filter_to_string(filter_type)}_{stacking_method}"
#         img_key_json = f"{filter_to_string(filter_type)}_{stacking_method}_info"
#         stack_dict[img_key] = stacking_outputs[(filter_type, stacking_method)][0]
#         stack_dict[img_key_json] = stacking_outputs[(filter_type, stacking_method)][1]
#
#     with open(stackinfo_output_path, "w") as f:
#         json.dump(stack_dict, f)


# def stacked_images_from_stackinfo(stackinfo_path: pathlib.Path) -> Dict:
#     with open(stackinfo_path, "r") as f:
#         stackinfo_dict = json.load(f)
#
#     image_dict = {}
#     filter_types = [SwiftFilter.uvv, SwiftFilter.uw1]
#     stacking_methods = [StackingMethod.summation, StackingMethod.median]
#     containing_folder = stackinfo_path.parent
#
#     for filter_type, stacking_method in itertools.product(
#         filter_types, stacking_methods
#     ):
#         # this is the key format of the stackinfo dict
#         img_key = f"{filter_to_string(filter_type)}_{stacking_method}"
#         img_key_json = f"{filter_to_string(filter_type)}_{stacking_method}_info"
#         # get paths of image and its json info
#         sip = containing_folder / pathlib.Path(stackinfo_dict[img_key])
#         siip = containing_folder / pathlib.Path(stackinfo_dict[img_key_json])
#         # reconstruct the stacked images from the FITS and JSON info
#         image_dict[(filter_type, stacking_method)] = read_stacked_image(
#             stacked_image_path=sip, stacked_image_info_path=siip
#         )
#
#     return image_dict
