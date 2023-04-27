import yaml
import pathlib
import logging as log
from typing import Optional


def read_swift_config(filepath: pathlib.Path) -> Optional[dict]:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml
