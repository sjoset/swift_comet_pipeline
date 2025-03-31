import pathlib


def get_long_period_comet_configs(base_dir_path: pathlib.Path) -> list[pathlib.Path]:

    long_period_prefix = "C_"
    long_period_directories = [
        d
        for d in base_dir_path.iterdir()
        if d.is_dir() and d.name.startswith(long_period_prefix)
    ]

    config_file_path = pathlib.Path("config.yaml")
    config_paths = [d / config_file_path for d in long_period_directories]

    return config_paths
