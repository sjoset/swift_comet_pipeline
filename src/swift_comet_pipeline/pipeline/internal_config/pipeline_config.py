import os
import yaml
import pathlib
import logging as log
from dataclasses import dataclass


@dataclass
class SwiftPipelineConfig:
    # TODO: document these
    solar_spectrum_path: pathlib.Path
    effective_area_uw1_path: pathlib.Path
    effective_area_uvv_path: pathlib.Path
    oh_fluorescence_path: pathlib.Path


def _read_yaml(filepath: pathlib.Path) -> dict | None:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml


def read_swift_pipeline_config() -> SwiftPipelineConfig | None:
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
    )

    return pipeline_config
