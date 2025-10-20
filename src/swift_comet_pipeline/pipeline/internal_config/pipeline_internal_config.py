import os
import cattrs
import pathlib
from dataclasses import dataclass
from functools import cache

from swift_comet_pipeline.common.read_yaml import read_yaml
from swift_comet_pipeline.scp_types.primitive.uvot_filter import UvotFilter


# effective_area_uuu_path: "../../../../data/swift_effective_areas/swuuu_20041120v104.arf"
# effective_area_ubb_path: "../../../../data/swift_effective_areas/swubb_20041120v104.arf"
# effective_area_uvv_path: "../../../../data/swift_effective_areas/swuvv_20041120v104.arf"
# effective_area_uw1_path: "../../../../data/swift_effective_areas/swuw1_20041120v106.arf"
# effective_area_um2_path: "../../../../data/swift_effective_areas/swum2_20041120v105.arf"
# effective_area_white_path: "../../../../data/swift_effective_areas/swuwh_20041120v104.arf"
# effective_area_vgrism_path: "../../../../data/swift_effective_areas/swugv1000_20041120v101.arf"
# effective_area_ugrism_path: "../../../../data/swift_effective_areas/swugu0200_20041120v101.arf"

# TODO: migrate the contents of the yaml to a dictionary/function here


@dataclass
class SwiftPipelineConfigYaml:
    effective_area_uuu_path: pathlib.Path
    effective_area_ubb_path: pathlib.Path
    effective_area_uvv_path: pathlib.Path
    effective_area_uw1_path: pathlib.Path
    effective_area_uw2_path: pathlib.Path
    effective_area_um2_path: pathlib.Path
    effective_area_white_path: pathlib.Path
    effective_area_vgrism_path: pathlib.Path
    effective_area_ugrism_path: pathlib.Path

    uvot_sensitivity_path: pathlib.Path
    oh_fluorescence_path: pathlib.Path


@dataclass
class SwiftPipelineConfig:
    uvot_sensitivity_path: pathlib.Path
    oh_fluorescence_path: pathlib.Path

    effective_areas: dict[UvotFilter, pathlib.Path]


@cache
def read_swift_pipeline_config() -> SwiftPipelineConfig | None:
    script_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))
    config_yaml = read_yaml(script_path / pathlib.Path("pipeline_internal_config.yaml"))

    if config_yaml is None:
        return None

    spcy = cattrs.structure(obj=config_yaml, cl=SwiftPipelineConfigYaml)

    filter_path_dict_str = {
        UvotFilter.uuu: spcy.effective_area_uuu_path,
        UvotFilter.ubb: spcy.effective_area_ubb_path,
        UvotFilter.uvv: spcy.effective_area_uvv_path,
        UvotFilter.uw1: spcy.effective_area_uw1_path,
        UvotFilter.uw2: spcy.effective_area_uw2_path,
        UvotFilter.um2: spcy.effective_area_um2_path,
        UvotFilter.white: spcy.effective_area_white_path,
        UvotFilter.vgrism: spcy.effective_area_vgrism_path,
        UvotFilter.ugrism: spcy.effective_area_ugrism_path,
    }
    filter_path_dict = {
        k: (script_path / pathlib.Path(v)).expanduser().resolve()
        for k, v in filter_path_dict_str.items()
    }

    spc = SwiftPipelineConfig(
        uvot_sensitivity_path=(script_path / spcy.uvot_sensitivity_path)
        .expanduser()
        .resolve(),
        oh_fluorescence_path=(script_path / spcy.oh_fluorescence_path)
        .expanduser()
        .resolve(),
        effective_areas=filter_path_dict,
    )

    return spc
