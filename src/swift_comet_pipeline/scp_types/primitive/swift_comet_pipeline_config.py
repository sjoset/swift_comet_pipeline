import pathlib
from dataclasses import dataclass


@dataclass
class SwiftCometPipelineConfig:
    solar_spectrum_path: pathlib.Path
    effective_area_uw1_path: pathlib.Path
    effective_area_uvv_path: pathlib.Path
    oh_fluorescence_path: pathlib.Path
    vectorial_model_path: pathlib.Path
