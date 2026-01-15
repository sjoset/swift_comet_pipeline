from dataclasses import asdict, replace
from functools import partial

import numpy as np
import astropy.units as u

from swift_comet_pipeline.pipeline.product_system.registry_and_store import Products
from swift_comet_pipeline.post_processing.fixed_aperture_size import (
    assemble_fixed_aperture_averaged_results,
    bayesian_expectation_of_assembled_fixed_aperture_results,
)
from swift_comet_pipeline.post_processing.vectorial_model_results import (
    assemble_vectorial_model_bayesian_production_rates,
    assemble_vectorial_model_production_rates,
    assemble_vectorial_model_results,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.compound.lightcurve import (
    LightCurve,
    LightCurveEntry,
    LightCurveEntrySource,
)
from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)
from swift_comet_pipeline.scp_types.primitive import *


# TODO: move these somewhere else
_fixed_aperture_radius = 100000 * u.km
_fixed_aperture_window = 10000 * u.km


def build_water_production_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    dust_redness_mean: DustReddeningPercent,
    dust_redness_sigma: DustReddeningPercent,
    epochs_to_include: EpochIndex | None = None,
) -> LightCurve:

    if not epochs_to_include:
        eids = scp.epoch_index
    else:
        eids = epochs_to_include
    assert eids is not None

    water_production_lightcurve: LightCurve = []

    for eid in eids:

        lightcurve_constructor = partial(
            LightCurveEntry,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            dust_redness=dust_redness_mean,
            **asdict(eid),
        )

        # get vectorial water production for this mean redness
        vdf = assemble_vectorial_model_results(
            scp=scp,
            eid=eid,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
        )
        vqdf = assemble_vectorial_model_production_rates(
            df=vdf, oh_filter=oh_filter, dust_filter=dust_filter
        )
        bqdf = assemble_vectorial_model_bayesian_production_rates(
            df=vqdf, dust_redness_sigma=dust_redness_sigma
        )
        vectorial_q = bqdf[
            bqdf.dust_redness_pct_per_hundred_nm == dust_redness_mean
        ].bayes_far_fit_q.iloc[0]
        vectorial_q_err = bqdf[
            bqdf.dust_redness_pct_per_hundred_nm == dust_redness_mean
        ].bayes_far_fit_q_err.iloc[0]

        vec_lc_entry = lightcurve_constructor(
            q=vectorial_q,
            q_err=vectorial_q_err,
            q_source=LightCurveEntrySource.from_vectorial,
        )

        vec_lc_entry = replace(
            vec_lc_entry,
            rh_au=vec_lc_entry.rh_au
            * np.sign(vec_lc_entry.time_from_perihelion.to_value(u.day)),  # type: ignore
        )
        water_production_lightcurve.append(vec_lc_entry)

        print(f"Assembling aperture results for {eid.epoch_id} ...")
        apdf = assemble_fixed_aperture_averaged_results(
            scp=scp,
            eid=eid,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            fixed_aperture_radius=_fixed_aperture_radius,
            fixed_aperture_window_lower=_fixed_aperture_window,
            fixed_aperture_window_upper=_fixed_aperture_window,
        )
        aperture_water_production_column = "aperture_matched_q_h2o_median"
        aperture_water_production_err_column = "aperture_matched_q_h2o_median_err"
        negative_production_included = (
            lambda x: x[aperture_water_production_column] > -np.inf
        )
        bapdf = bayesian_expectation_of_assembled_fixed_aperture_results(
            df=apdf,
            dust_redness_mean=dust_redness_mean,
            dust_redness_sigma=dust_redness_sigma,
            constraint_function=negative_production_included,
        )

        ap_q = bapdf[aperture_water_production_column].iloc[0]
        ap_q_err = bapdf[aperture_water_production_err_column].iloc[0]

        ap_lc_entry = lightcurve_constructor(
            q=ap_q, q_err=ap_q_err, q_source=LightCurveEntrySource.from_aperture
        )
        ap_lc_entry = replace(
            ap_lc_entry,
            rh_au=ap_lc_entry.rh_au
            * np.sign(ap_lc_entry.time_from_perihelion.to_value(u.day)),  # type: ignore
        )
        water_production_lightcurve.append(ap_lc_entry)

    return water_production_lightcurve
