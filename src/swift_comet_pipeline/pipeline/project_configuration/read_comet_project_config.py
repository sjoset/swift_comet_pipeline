import pathlib

from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive import *

from swift_comet_pipeline.common.read_yaml import read_yaml


# TODO: remove old code
# def _path_from_yaml(yaml_dict: dict, key: str) -> pathlib.Path | None:
#     """
#     Extracts the string yaml_dict[key], and if it exists, turn it into a pathlib.Path, expands and resolves the path
#     """
#
#     val = yaml_dict.get(key, None)
#     if val is not None:
#         val = pathlib.Path(val).expanduser().resolve()
#
#     return val


# TODO: remove old code
def read_comet_project_config(
    config_path: pathlib.Path,
) -> CometProjectConfig | None:
    """
    Returns a SwiftProjectConfig given the yaml config file path, filling in optional values with defaults
    """
    config_yaml = read_yaml(config_path)
    if config_yaml is None:
        return None

    project_config = cattrs.structure(obj=config_yaml, cl=CometProjectConfig)

    # TODO: do validation here, including checking the maximum/minimum dust values fall within the range that the filters allow
    # that keeps beta finite
    return project_config

    # swift_data_path = _path_from_yaml(config_yaml, "swift_data_path")
    # project_path = _path_from_yaml(config_yaml, "project_path")
    # if swift_data_path is None or project_path is None:
    #     print(
    #         f"Could not find necessary entries: swift_data_path or project_path in {config_path}"
    #     )
    #     return None
    #
    # jpl_horizons_id = config_yaml.get("jpl_horizons_id", None)
    # if jpl_horizons_id is None:
    #     print(f"Could not find jpl_horizons_id in {config_path}")
    #     return None
    #
    # project_config = CometProjectConfig(
    #     swift_data_path=swift_data_path,
    #     jpl_horizons_id=jpl_horizons_id,
    #     project_path=project_path,
    #     vectorial_model_quality=VectorialModelGridQuality(
    #         config_yaml["vectorial_model_quality"]
    #     ),
    #     vectorial_model_backend=VectorialModelBackend(
    #         config_yaml["vectorial_model_backend"]
    #     ),
    #     vectorial_fitting_requires_km=float(
    #         config_yaml.get("vectorial_fitting_requires_km", 100_000)
    #     ),
    #     near_far_split_radius_km=float(
    #         config_yaml.get("near_far_split_radius_km", 50_000)
    #     ),
    # )
    # return project_config


# def read_swift_comet_project_config(
#     config_path: pathlib.Path,
# ) -> CometProjectConfig | None:
#     """
#     Returns a SwiftProjectConfig given the yaml config file path, filling in optional values with defaults
#     """
#     config_yaml = read_yaml(config_path)
#     if config_yaml is None:
#         return None
#
#     swift_data_path = _path_from_yaml(config_yaml, "swift_data_path")
#     project_path = _path_from_yaml(config_yaml, "project_path")
#     if swift_data_path is None or project_path is None:
#         print(
#             f"Could not find necessary entries: swift_data_path or project_path in {config_path}"
#         )
#         return None
#     jpl_horizons_id = config_yaml.get("jpl_horizons_id", None)
#     if jpl_horizons_id is None:
#         print(f"Could not find jpl_horizons_id in {config_path}")
#         return None
#
#     project_config = CometProjectConfig(
#         swift_data_path=swift_data_path,
#         jpl_horizons_id=jpl_horizons_id,
#         project_path=project_path,
#         vectorial_model_quality=VectorialModelGridQuality(
#             config_yaml["vectorial_model_quality"]
#         ),
#         vectorial_model_backend=VectorialModelBackend(
#             config_yaml["vectorial_model_backend"]
#         ),
#         vectorial_fitting_requires_km=float(
#             config_yaml.get("vectorial_fitting_requires_km", 100_000)
#         ),
#         near_far_split_radius_km=float(
#             config_yaml.get("near_far_split_radius_km", 50_000)
#         ),
#     )
#     return project_config
