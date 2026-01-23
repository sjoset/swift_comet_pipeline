from dataclasses import asdict, replace
from functools import partial

import numpy as np
import astropy.units as u

from swift_comet_pipeline.modeling.vectorial.molecular_parameters import (
    make_hydroxyl_fragment,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.post_processing.blue_spot import (
    bayes_blue_spot_df_from_blue_spot_df,
    blue_spot_df_from_vectorial_df,
)
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


# # TODO: move these somewhere else
# _fixed_aperture_radius = 100000 * u.km  # type: ignore
# _fixed_aperture_window = 10000 * u.km  # type: ignore
#
#
# # TODO: maybe break this into non-bayesian & bayesian versions?
# def build_water_production_lightcurve(
#     scp: Products,
#     oh_filter: UvotFilter,
#     dust_filter: UvotFilter,
#     stacking_method: StackingMethod,
#     dust_redness_mean: DustReddeningPercent,
#     dust_redness_sigma: DustReddeningPercent,
#     epochs_to_include: EpochIndex | None = None,
# ) -> LightCurve:
#     """
#     Aperture and vectorial model water production rates as a LightCurve,
#     using the filter pair for continuum subtraction, images from the given stacking method, and applying
#     Bayesian color analysis with the given redness distribution mean/sigma
#     """
#
#     if not epochs_to_include:
#         eids = scp.epoch_index
#     else:
#         eids = epochs_to_include
#     assert eids is not None
#
#     water_production_lightcurve: LightCurve = []
#
#     for eid in eids:
#
#         lightcurve_constructor = partial(
#             LightCurveEntry,
#             oh_filter=oh_filter,
#             dust_filter=dust_filter,
#             stacking_method=stacking_method,
#             dust_redness=dust_redness_mean,
#             **asdict(eid),
#         )
#
#         # get vectorial water production for this mean redness
#         vdf = assemble_vectorial_model_results(
#             scp=scp,
#             eid=eid,
#             oh_filter=oh_filter,
#             dust_filter=dust_filter,
#             stacking_method=stacking_method,
#         )
#         vqdf = assemble_vectorial_model_production_rates(
#             df=vdf, oh_filter=oh_filter, dust_filter=dust_filter
#         )
#         bqdf = assemble_vectorial_model_bayesian_production_rates(
#             df=vqdf, dust_redness_sigma=dust_redness_sigma
#         )
#         vectorial_q = bqdf[
#             bqdf.dust_redness_pct_per_hundred_nm == dust_redness_mean
#         ].bayes_far_fit_q.iloc[0]
#         vectorial_q_err = bqdf[
#             bqdf.dust_redness_pct_per_hundred_nm == dust_redness_mean
#         ].bayes_far_fit_q_err.iloc[0]
#
#         vec_lc_entry = lightcurve_constructor(
#             q_h2o=vectorial_q,
#             q_h2o_err=vectorial_q_err,
#             q_source=LightCurveEntrySource.from_vectorial,
#         )
#
#         vec_lc_entry = replace(
#             vec_lc_entry,
#             rh_au=vec_lc_entry.rh_au
#             * np.sign(vec_lc_entry.time_from_perihelion.to_value(u.day)),  # type: ignore
#         )
#         water_production_lightcurve.append(vec_lc_entry)
#
#         print(f"Assembling aperture results for {eid.epoch_id} ...")
#         apdf = assemble_fixed_aperture_averaged_results(
#             scp=scp,
#             eid=eid,
#             oh_filter=oh_filter,
#             dust_filter=dust_filter,
#             stacking_method=stacking_method,
#             fixed_aperture_radius=_fixed_aperture_radius,
#             fixed_aperture_window_lower=_fixed_aperture_window,
#             fixed_aperture_window_upper=_fixed_aperture_window,
#         )
#         aperture_water_production_column = "aperture_matched_q_h2o_median"
#         aperture_water_production_err_column = "aperture_matched_q_h2o_median_err"
#         aperture_hydroxyl_production_column = "q_oh_median"
#         aperture_hydroxyl_production_err_column = "q_oh_median_err"
#         negative_production_included = (
#             lambda x: x[aperture_water_production_column] > -np.inf
#         )
#         bapdf = bayesian_expectation_of_assembled_fixed_aperture_results(
#             df=apdf,
#             dust_redness_mean=dust_redness_mean,
#             dust_redness_sigma=dust_redness_sigma,
#             constraint_function=negative_production_included,
#         )
#
#         ap_q_h2o = bapdf[aperture_water_production_column].iloc[0]
#         ap_q_h2o_err = bapdf[aperture_water_production_err_column].iloc[0]
#         ap_q_oh = bapdf[aperture_hydroxyl_production_column].iloc[0]
#         ap_q_oh_err = bapdf[aperture_hydroxyl_production_err_column].iloc[0]
#
#         ap_lc_entry = lightcurve_constructor(
#             q_h2o=ap_q_h2o,
#             q_h2o_err=ap_q_h2o_err,
#             q_oh=ap_q_oh,
#             q_oh_err=ap_q_oh_err,
#             q_source=LightCurveEntrySource.from_aperture,
#         )
#         ap_lc_entry = replace(
#             ap_lc_entry,
#             rh_au=ap_lc_entry.rh_au
#             * np.sign(ap_lc_entry.time_from_perihelion.to_value(u.day)),  # type: ignore
#         )
#         water_production_lightcurve.append(ap_lc_entry)
#
#     return water_production_lightcurve


def build_blue_spot_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    dust_redness_mean: DustReddeningPercent,
    dust_redness_sigma: DustReddeningPercent,
    blue_spot_extent_km: float,
    epochs_to_include: EpochIndex | None = None,
) -> LightCurve:
    """
    Aperture and vectorial model water production rates as a LightCurve,
    using the filter pair for continuum subtraction, images from the given stacking method, and applying
    Bayesian color analysis with the given redness distribution mean/sigma
    """

    if not epochs_to_include:
        eids = scp.epoch_index
    else:
        eids = epochs_to_include
    assert eids is not None

    blue_spot_lightcurve: LightCurve = []

    for eid in eids:

        if eid.rh_au > 3.0:
            print(
                f"Skipping {eid.epoch_id}: rh of {eid.rh_au} is too large for blue spot analysis."
            )
            continue

        oh_fragment = make_hydroxyl_fragment()
        scaled_oh_lifetime = oh_fragment.tau_T_s * eid.rh_au**2

        lightcurve_constructor = partial(
            LightCurveEntry,
            q_h2o=0.0,
            q_h2o_err=0.0,
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
        bsdf = blue_spot_df_from_vectorial_df(
            df=vdf,
            scaled_oh_lifetime=scaled_oh_lifetime,
            blue_spot_extent_km=blue_spot_extent_km,
        )
        bayes_bsdf = bayes_blue_spot_df_from_blue_spot_df(
            df=bsdf, dust_redness_sigma=dust_redness_sigma
        )
        blue_spot_q_oh = bayes_bsdf[
            bayes_bsdf.dust_redness_pct_per_hundred_nm == dust_redness_mean
        ].bayes_blue_spot_q_oh
        # TODO: figure out reasonable error for this calculation
        blue_spot_q_oh_err = 0

        bayes_blue_spot_lc_entry = lightcurve_constructor(
            q_oh=blue_spot_q_oh,
            q_oh_err=blue_spot_q_oh_err,
            q_source=LightCurveEntrySource.from_blue_spot,
        )
        # apply sign to rh_au for pre-perihelion
        bayes_blue_spot_lc_entry = replace(
            bayes_blue_spot_lc_entry,
            rh_au=bayes_blue_spot_lc_entry.rh_au
            * np.sign(bayes_blue_spot_lc_entry.time_from_perihelion.to_value(u.day)),  # type: ignore
        )
        blue_spot_lightcurve.append(bayes_blue_spot_lc_entry)

    return blue_spot_lightcurve
