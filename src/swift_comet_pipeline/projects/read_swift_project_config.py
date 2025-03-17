import yaml
import pathlib
import logging as log

from swift_comet_pipeline.modeling.vectorial_model_backend import VectorialModelBackend
from swift_comet_pipeline.modeling.vectorial_model_grid import VectorialModelGridQuality
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


# TODO: this function is defined multiple times in different files: move it into its own file somewhere
def _read_yaml(filepath: pathlib.Path) -> dict | None:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml


def _path_from_yaml(yaml_dict: dict, key: str) -> pathlib.Path | None:
    """
    Extracts the string yaml_dict[key], and if it exists, turn it into a pathlib.Path
    """

    val = yaml_dict.get(key, None)
    if val is not None:
        val = pathlib.Path(val).expanduser().resolve()

    return val


def read_swift_project_config(
    config_path: pathlib.Path,
) -> SwiftProjectConfig | None:
    """
    Returns a SwiftProjectConfig given the yaml config file path, filling in optional values with defaults
    """
    config_yaml = _read_yaml(config_path)
    if config_yaml is None:
        return None

    swift_data_path = _path_from_yaml(config_yaml, "swift_data_path")
    project_path = _path_from_yaml(config_yaml, "project_path")
    if swift_data_path is None or project_path is None:
        print(
            f"Could not find necessary entries: swift_data_path or project_path in {config_path}"
        )
        return None
    jpl_horizons_id = config_yaml.get("jpl_horizons_id", None)
    if jpl_horizons_id is None:
        print(f"Could not find jpl_horizons_id in {config_path}")
        return None

    project_config = SwiftProjectConfig(
        swift_data_path=swift_data_path,
        jpl_horizons_id=jpl_horizons_id,
        project_path=project_path,
        vectorial_model_quality=VectorialModelGridQuality(
            config_yaml["vectorial_model_quality"]
        ),
        vectorial_model_backend=VectorialModelBackend(
            config_yaml["vectorial_model_backend"]
        ),
        vectorial_fitting_requires_km=float(
            config_yaml.get("vectorial_fitting_requires_km", 100_000)
        ),
        near_far_split_radius_km=float(
            config_yaml.get("near_far_split_radius_km", 50_000)
        ),
    )
    return project_config
