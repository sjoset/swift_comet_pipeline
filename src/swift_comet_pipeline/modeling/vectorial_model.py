from functools import cache
from typing import Callable

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
from swift_comet_pipeline.types.swift_project_config import SwiftProjectConfig


def vectorial_model_settings_init(
    swift_project_config: SwiftProjectConfig,
) -> None:
    vectorial_model_cache_init(project_path=swift_project_config.project_path)
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
    # TODO: water grains modeling as slow-moving water parents did not work - remove code or try something else

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


def num_OH_from_vectorial_model_result(
    vmr: VectorialModelResult,
) -> float:
    """
    Counts *all* of the OH molecules from a given vectorial model, out to the maximum r from the comet nucleus
    that was modeled
    """
    # TODO: come up with a decent error range and change return value to HydroxylMoleculeCount

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


@u.quantity_input
def num_OH_from_vectorial_model_result_within_r(
    vmr: VectorialModelResult, within_r: u.Quantity[u.m]  # type: ignore
) -> float:
    """
    Counts OH molecules within 'within_r' distance of the nucleus by integrating the column density in a circle of radius 'within_r'
    """
    # TODO: come up with a decent error range and change return value to HydroxylMoleculeCount

    # total fragments = circular area integral of theta [0, 2*pi], [0, r_end] of CD(r) r dr dtheta = 2 pi * int( CD(r) * r dr )
    def integrand(r_m: float, column_density_interpolation_function: Callable):
        return column_density_interpolation_function(r_m) * r_m

    model_min_r_m = np.min(vmr.column_density_grid.to(u.m).value)  # type: ignore
    model_max_r_m = np.max(vmr.column_density_grid.to(u.m).value)  # type: ignore
    within_r_m = within_r.to(u.m).value  # type: ignore

    r_begin = model_min_r_m
    r_end = min(within_r_m, model_max_r_m)

    num_oh_r, _ = quad(
        integrand, a=r_begin, b=r_end, args=(vmr.column_density_interpolation,)
    )

    num_oh = 2 * np.pi * num_oh_r

    return num_oh
