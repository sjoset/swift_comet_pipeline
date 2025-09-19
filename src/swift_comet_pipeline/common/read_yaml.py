import pathlib
import yaml


# TODO: re-do the logging for errors below
def read_yaml(filepath: pathlib.Path) -> dict | None:
    """Read YAML file from disk and return dictionary with the contents"""
    with open(filepath, "r") as stream:
        try:
            param_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            param_yaml = None
            print(exc)
            # log.info("Reading file %s resulted in yaml error: %s", filepath, exc)

    return param_yaml
