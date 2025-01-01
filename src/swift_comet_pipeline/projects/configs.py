import yaml
import pathlib
import logging as log
from dataclasses import asdict

from rich import print as rprint

from swift_comet_pipeline.modeling.vectorial_model_backend import VectorialModelBackend
from swift_comet_pipeline.modeling.vectorial_model_grid import VectorialModelGridQuality
from swift_comet_pipeline.projects.swift_project_config import SwiftProjectConfig
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.tui.tui_common import get_yes_no


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


def read_or_create_project_config(
    swift_project_config_path: pathlib.Path,
) -> SwiftProjectConfig | None:
    # check if project config exists, and offer to create if not
    if not swift_project_config_path.exists():
        print(
            f"Config file {swift_project_config_path} does not exist! Would you like to create one now? (y/n)"
        )
        create_config = get_yes_no()
        if create_config:
            create_swift_project_config_from_input(
                swift_project_config_path=swift_project_config_path
            )
        else:
            return

    # load the project config
    swift_project_config = read_swift_project_config(swift_project_config_path)
    if swift_project_config is None:
        print(f"Error reading config file {swift_project_config_path}, exiting.")
        return None

    return swift_project_config


def create_swift_project_config_from_input(
    swift_project_config_path: pathlib.Path,
) -> None:
    """
    Collect info on the data directories and how to identify the comet through JPL horizons,
    and write it to a yaml config
    """

    swift_data_path = pathlib.Path(input("Directory of the downloaded swift data: "))

    # try to validate that this path actually has data before accepting
    test_of_swift_data = SwiftData(data_path=swift_data_path)
    num_obsids = len(test_of_swift_data.get_all_observation_ids())
    if num_obsids == 0:
        rprint(
            "There doesn't seem to be data in the necessary format at [blue]{swift_data_path}[/blue]!"
        )
    else:
        rprint(
            f"Found appropriate data with a total of [green]{num_obsids}[/green] observation IDs"
        )

    project_path = pathlib.Path(
        input("Directory to store results and intermediate products: ")
    )

    jpl_horizons_id = input("JPL Horizons ID of the comet: ")

    # TODO: this fails on invalid input, make it more robust
    vm_quality = input(
        f"Vectorial model quality {VectorialModelGridQuality.all_qualities()}: "
    )

    # TODO: this fails on invalid input, make it more robust
    vm_backend = input(
        f"Vectorial model backend {VectorialModelBackend.all_model_backends()}: "
    )

    # TODO: finish questions for vectorial_fitting_requires_km and near_far_split_radius_km

    swift_project_config = SwiftProjectConfig(
        swift_data_path=swift_data_path,
        jpl_horizons_id=jpl_horizons_id,
        project_path=project_path,
        vectorial_model_quality=VectorialModelGridQuality(vm_quality),
        vectorial_model_backend=VectorialModelBackend(vm_backend),
        vectorial_fitting_requires_km=float(100_000),
        near_far_split_radius_km=float(50_000),
    )

    print(f"Writing project config to {swift_project_config_path}...")
    write_swift_project_config(
        config_path=swift_project_config_path, swift_project_config=swift_project_config
    )
