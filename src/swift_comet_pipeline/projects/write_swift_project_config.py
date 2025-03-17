import yaml
import pathlib
from dataclasses import asdict

from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


def write_swift_project_config(
    config_path: pathlib.Path, swift_project_config: SwiftProjectConfig
) -> None:
    dict_to_write = asdict(swift_project_config)

    dict_to_write["vectorial_model_quality"] = str(
        dict_to_write["vectorial_model_quality"]
    )

    with open(config_path, "w") as stream:
        try:
            yaml.safe_dump(dict_to_write, stream)
        except yaml.YAMLError as exc:
            print(exc)
