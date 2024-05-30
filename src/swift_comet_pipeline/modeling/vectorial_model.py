import pathlib
from functools import cache

import numpy as np
from icecream import ic

import sbpy.activity as sba
import astropy.units as u

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
    run_vectorial_models,
)

from swift_comet_pipeline.modeling.molecular_parameters import (
    make_hydroxyl_fragment,
    make_slow_water_molecule_parent,
    make_water_molecule_parent,
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


@cache
@u.quantity_input
def num_OH_at_r_au_vectorial(
    base_q: u.Quantity[1 / u.s],  # type: ignore
    helio_r: u.Quantity[u.AU],  # type: ignore
    water_grains: bool = False,
) -> tuple[float, sba.VectorialModel]:

    vmcache_path = get_vectorial_model_cache_path()

    # modeling parameters
    production = CometProduction(base_q_per_s=base_q.to(1 / u.s).value)
    water_parent = make_water_molecule_parent()
    hydroxyl_fragment = make_hydroxyl_fragment()
    grid = make_vectorial_model_grid()
    untransformed_vmc = VectorialModelConfig(
        production=production,
        parent=water_parent,
        fragment=hydroxyl_fragment,
        grid=grid,
    )

    # scale the molecular parameters for r_h
    vmc = apply_input_transform(
        vmc=untransformed_vmc, r_h=helio_r, xfrm=VmcTransform.cochran_schleicher_93
    )
    if water_grains:
        new_parent = make_slow_water_molecule_parent(
            v_outflow=(0.01 * u.km / u.s) / np.sqrt(helio_r.to(u.AU).value)
        )
        vmc = VectorialModelConfig(
            production=vmc.production,
            parent=new_parent,
            fragment=vmc.fragment,
            grid=vmc.grid,
        )
    vmcalculation = run_vectorial_models(
        vmc_list=[vmc],
        vectorial_model_cache_path=vmcache_path,
    )[0]

    coma = vmcalculation.vmr.coma
    if coma is None:
        ic("Vectorial model calculation results are empty! This is a bug!")
        ic(vmcalculation)
        exit(1)

    return (coma.vmr.num_fragments_grid, coma)
