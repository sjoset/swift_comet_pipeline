from functools import cache
from typing import Callable

# from icecream import ic
import numpy as np
import astropy.units as u

from pyvectorial_au.model_output.vectorial_model_result import VectorialModelResult
from pyvectorial_au.model_input.vectorial_model_config import (
    CometProduction,
    VectorialModelConfig,
)
from pyvectorial_au.backends.rust_version import RustModelExtraConfig
from pyvectorial_au.pre_model_processing.input_transforms import (
    VmcTransform,
    apply_input_transform,
)
from pyvectorial_au.model_running.vectorial_model_runner import (
    PythonModelExtraConfig,
    run_vectorial_models,
)
from scipy.integrate import quad

from swift_comet_pipeline.modeling.molecular_parameters import (
    make_hydroxyl_fragment,
    make_slow_water_molecule_parent,
    make_water_molecule_parent,
)
from swift_comet_pipeline.modeling.vectorial_model_backend import (
    VectorialModelBackend,
    get_vectorial_model_backend,
    vectorial_model_backend_init,
)
from swift_comet_pipeline.modeling.vectorial_model_cache import (
    get_vectorial_model_cache_path,
    vectorial_model_cache_init,
)
from swift_comet_pipeline.modeling.vectorial_model_grid import (
    make_vectorial_model_grid,
    vectorial_model_grid_quality_init,
)
from swift_comet_pipeline.projects.configs import SwiftProjectConfig


def vectorial_model_settings_init(
    swift_project_config: SwiftProjectConfig,
) -> None:
    vectorial_model_cache_init(swift_project_config=swift_project_config)
    vectorial_model_grid_quality_init(
        quality=swift_project_config.vectorial_model_quality
    )
    vectorial_model_backend_init(backend=swift_project_config.vectorial_model_backend)


@cache
@u.quantity_input
def water_vectorial_model(
    base_q: u.Quantity[1 / u.s],  # type: ignore
    helio_r: u.Quantity[u.AU],  # type: ignore
    water_grains: bool = False,
) -> VectorialModelResult:
    vmcache_path = get_vectorial_model_cache_path()

    # modeling parameters
    production = CometProduction(base_q_per_s=base_q.to(1 / u.s).value)  # type: ignore
    water_parent = make_water_molecule_parent()
    hydroxyl_fragment = make_hydroxyl_fragment()
    grid = make_vectorial_model_grid()
    untransformed_vmc = VectorialModelConfig(
        production=production,
        parent=water_parent,
        fragment=hydroxyl_fragment,
        grid=grid,
    )

    # scale the molecular parameters for heliocentric distance
    vmc = apply_input_transform(
        vmc=untransformed_vmc, r_h=helio_r, xfrm=VmcTransform.cochran_schleicher_93
    )
    if water_grains:
        new_parent = make_slow_water_molecule_parent(
            v_outflow=(0.01 * u.km / u.s) / np.sqrt(helio_r.to(u.AU).value)  # type: ignore
        )
        vmc = VectorialModelConfig(
            production=vmc.production,
            parent=new_parent,
            fragment=vmc.fragment,
            grid=vmc.grid,
        )

    model_backend = get_vectorial_model_backend()
    # ic(f"Using model backend {model_backend}..")
    if model_backend == VectorialModelBackend.sbpy:
        extra_config = PythonModelExtraConfig(print_progress=False)
    elif model_backend == VectorialModelBackend.rust:
        extra_config = RustModelExtraConfig()
    else:
        print(f"Invalid model backend {model_backend} specified!  Defaulting to sbpy.")
        extra_config = PythonModelExtraConfig(print_progress=False)

    vmcalculation = run_vectorial_models(
        vmc_list=[vmc],
        vectorial_model_cache_path=vmcache_path,
        extra_config=extra_config,
    )[0]

    return vmcalculation.vmr


def num_OH_from_vectorial_model_result(vmr: VectorialModelResult) -> float:

    # total fragments = 4 * pi * integral of volume density at r * r^2
    def integrand(r_m: float, volume_density_interpolation_function: Callable):
        return volume_density_interpolation_function(r_m) * r_m**2

    r_begin = np.min(vmr.volume_density_grid.to(u.m).value)  # type: ignore
    r_end = np.max(vmr.volume_density_grid.to(u.m).value)  # type: ignore
    num_oh_r, _ = quad(
        integrand, a=r_begin, b=r_end, args=(vmr.volume_density_interpolation,)
    )

    num_oh = 4 * np.pi * num_oh_r

    return num_oh
