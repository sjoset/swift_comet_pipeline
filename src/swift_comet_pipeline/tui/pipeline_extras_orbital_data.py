# from rich import print as rprint
from swift_comet_pipeline.orbits.orbital_data_download import download_orbital_data
from swift_comet_pipeline.projects.configs import SwiftProjectConfig


def pipeline_extra_orbital_data(swift_project_config: SwiftProjectConfig) -> None:
    download_orbital_data(swift_project_config=swift_project_config)
