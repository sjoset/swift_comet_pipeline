import os
import yaml
import pathlib
import logging as log
from typing import Callable, Optional
from dataclasses import asdict, dataclass


__all__ = [
    "SwiftProjectConfig",
    "SwiftPipelineConfig",
    "read_swift_project_config",
    "read_swift_pipeline_config",
    "write_swift_project_config",
]


@dataclass
class SwiftProjectConfig:
    swift_data_path: pathlib.Path
    jpl_horizons_id: str
    product_save_path: pathlib.Path


@dataclass
class SwiftPipelineConfig:
    solar_spectrum_path: pathlib.Path
    effective_area_uw1_path: pathlib.Path
    effective_area_uvv_path: pathlib.Path
    oh_fluorescence_path: pathlib.Path
    vectorial_model_path: pathlib.Path


def _read_yaml(filepath: pathlib.Path) -> Optional[dict]:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml


def _path_from_yaml(yaml_dict: dict, key: str) -> Optional[pathlib.Path]:
    """
    Extracts the string yaml_dict[key], and if it exists, turn it into a pathlib.Path
    """

    val = yaml_dict.get(key, None)
    if val is not None:
        val = pathlib.Path(val).expanduser().resolve()

    return val


def read_swift_pipeline_config() -> Optional[SwiftPipelineConfig]:
    script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))
    config_yaml = _read_yaml(script_path / pathlib.Path("pipeline_config.yaml"))

    if config_yaml is None:
        return None

    pipeline_config = SwiftPipelineConfig(
        solar_spectrum_path=script_path
        / pathlib.Path(config_yaml["solar_spectrum_path"]),
        effective_area_uw1_path=script_path
        / pathlib.Path(config_yaml["effective_area_uw1_path"]),
        effective_area_uvv_path=script_path
        / pathlib.Path(config_yaml["effective_area_uvv_path"]),
        oh_fluorescence_path=script_path
        / pathlib.Path(config_yaml["oh_fluorescence_path"]),
        vectorial_model_path=script_path
        / pathlib.Path(config_yaml["vectorial_model_path"]),
    )

    return pipeline_config


def read_swift_project_config(
    config_path: pathlib.Path,
) -> Optional[SwiftProjectConfig]:
    """
    Returns a SwiftProjectConfig given the yaml config file path
    """
    config_yaml = _read_yaml(config_path)
    if config_yaml is None:
        return None

    swift_data_path = _path_from_yaml(config_yaml, "swift_data_path")
    product_save_path = _path_from_yaml(config_yaml, "product_save_path")
    if swift_data_path is None or product_save_path is None:
        print(
            f"Could not find necessary entries: swift_data_path or product_save_path in {config_path}"
        )
        return None
    jpl_horizons_id = config_yaml.get("jpl_horizons_id", None)
    if jpl_horizons_id is None:
        print(f"Could not find jpl_horizons_id in {config_path}")
        return None

    project_config = SwiftProjectConfig(
        swift_data_path=swift_data_path,
        jpl_horizons_id=jpl_horizons_id,
        product_save_path=product_save_path,
    )
    return project_config


def write_swift_project_config(
    config_path: pathlib.Path, swift_project_config: SwiftProjectConfig
) -> None:
    dict_to_write = asdict(swift_project_config)

    path_keys_to_convert = [
        "swift_data_path",
        "product_save_path",
    ]
    # TODO: convert_or_delete is from an older implementation where config values could have been missing,
    # but the config was re-worked and now we can just convert and assume the value are there without checking and deleting missing values
    for k in path_keys_to_convert:
        convert_or_delete(dict_to_write, k, os.fspath)

    with open(config_path, "w") as stream:
        try:
            yaml.safe_dump(dict_to_write, stream)
        except yaml.YAMLError as exc:
            print(exc)


def convert_or_delete(d: dict, k: str, conversion_function: Callable):
    """Applies f to members of the dictionary, but deletes the members whose value is None"""
    if d[k] is None:
        del d[k]
    else:
        d[k] = conversion_function(d[k])
