import astropy.units as u
import numpy as np

from swift_comet_pipeline.modeling.vectorial.vectorial_model import (
    water_vectorial_model,
)
from swift_comet_pipeline.modeling.vectorial.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.photometry.comet.calculate_column_density import (
    calculate_oh_column_density,
)
from swift_comet_pipeline.photometry.dust.reddening_translate import (
    recalculate_redness_with_new_filter_pair,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductReference,
    Products,
    ContinuumSubtractionKey,
)
from swift_comet_pipeline.scp_types.compound.radial_profile_water_production import (
    RadialProfileWaterProductionAnalysis,
)
from swift_comet_pipeline.scp_types.primitive.vectorial_model_fit_type import (
    VectorialFitType,
)
from swift_comet_pipeline.swift.filters.filter_wavelengths import (
    calculate_mid_wavelength_nm,
)


def do_radial_profile_water_production(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, ContinuumSubtractionKey)

    # load the epoch info
    eid = scp.load_epoch_index_entry(epoch_id=ref.key.epoch_id)
    assert eid is not None

    oh_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.oh_filter,
        stacking_method=ref.key.stacking_method,
    )
    dust_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
    )

    # The product key holds the redness at a certain reference mid wavelength - convert the redness to the proper value
    # for the filters in use while we do the calculations
    # All of the products are stored on disk referring to the dust redness relative to redness_mid_wavelength_nm in the config file
    untransformed_redness = ref.key.dust_redness_pct_per_hundred_nm
    from_mid_wavelength = scp.cfg.redness_mid_wavelength_nm
    to_mid_wavelength = calculate_mid_wavelength_nm(
        filter_one=oh_key.filter_type, filter_two=dust_key.filter_type
    )
    transformed_dust_redness = np.round(
        recalculate_redness_with_new_filter_pair(
            known_redness=untransformed_redness,
            old_mid_wave_nm=from_mid_wavelength,
            new_filter_one=oh_key.filter_type,
            new_filter_two=dust_key.filter_type,
        )
    )
    print(
        f"Transforming given redness of {untransformed_redness} at mid wave {from_mid_wavelength} to {transformed_dust_redness} at mid wave {to_mid_wavelength}."
    )

    # load the radial profiles from the oh and dust filters
    oh_rad_prof = scp.load_extracted_radial_profile(key=oh_key)
    dust_rad_prof = scp.load_extracted_radial_profile(key=dust_key)

    # TODO: this should carry error!
    # do continuum subtraction to isolate oh column density
    oh_col_dens = calculate_oh_column_density(
        eid=eid,
        oh_profile=oh_rad_prof,
        dust_profile=dust_rad_prof,
        # dust_redness=ref.key.dust_redness_pct_per_hundred_nm,
        dust_redness=transformed_dust_redness,
        oh_filter=oh_key.filter_type,
        dust_filter=dust_key.filter_type,
    )

    model_Q = 1e28 / u.s
    rh = eid.rh_au * u.AU  # type: ignore
    # run vectorial model
    vmr = water_vectorial_model(base_q=model_Q, helio_r=rh)

    # fit column density to vectorial model along different regions: only near-nucleus data, data far from nucleus, and full curve fit
    col_dens_min_r_km = max(1, min(oh_col_dens.rs_km))
    near_far_split_km = scp.cfg.near_far_split_radius_km
    col_dens_max_r_km = max(oh_col_dens.rs_km)

    vectorial_fitting_bounds = {
        VectorialFitType.near_fit: (col_dens_min_r_km, near_far_split_km),
        VectorialFitType.far_fit: (near_far_split_km, col_dens_max_r_km),
        VectorialFitType.full_fit: (col_dens_min_r_km, col_dens_max_r_km),
    }
    vectorial_fits = {
        fit_type: vectorial_fit(
            fragment_column_density=oh_col_dens,
            vmr=vmr,
            model_Q=model_Q,
            r_fit_min=bound[0] * u.km,  # type: ignore
            r_fit_max=bound[1] * u.km,  # type: ignore
        )
        for fit_type, bound in vectorial_fitting_bounds.items()
    }

    rpwpa = RadialProfileWaterProductionAnalysis(
        oh_column_density=oh_col_dens, **vectorial_fits
    )

    # write results
    scp.save_radial_profile_water_production_analysis(rpwpa=rpwpa, key=ref.key)
