from dataclasses import replace
from functools import partial

from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import (
    BayesianPriorLightcurveKey,
    LightcurveKey,
)
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.pipeline.water_production_filter_pairs import (
    get_valid_water_production_filter_pairs,
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
from swift_comet_pipeline.scp_types.compound.water_production_filter_pair import (
    WaterProductionFilterPair,
)
from swift_comet_pipeline.scp_types.primitive import *


# TODO: priority 1: move these somewhere else - into the project config?
_fixed_aperture_radius = 100000 * u.km  # type: ignore
_fixed_aperture_window = 10000 * u.km  # type: ignore


# non-bayesian version
def build_water_production_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    epochs_to_include: EpochIndex | None = None,
) -> LightCurve:
    """
    Aperture and vectorial model water production rates as a LightCurve,
    using the filter pair for continuum subtraction, images from the given stacking method,
    and returns LightCurve object with both vectorial far fit water production and
    aperture water production, along with aperture-derived Q(OH).

    The returned lightcurve is a mix of all rednesses and 'q_source' in LightCurveEntry,
    and no Bayesian color analysis is applied.
    """

    if not epochs_to_include:
        eids = scp.epoch_index
    else:
        eids = epochs_to_include
    assert eids is not None

    current_filter_pair = WaterProductionFilterPair(
        oh_filter=oh_filter, dust_filter=dust_filter
    )
    water_production_lightcurve: LightCurve = []

    for eid in eids:

        # do we have observations in the filter pair for this epoch?
        valid_filter_pairs = get_valid_water_production_filter_pairs(
            eid=eid, oh_filters=[oh_filter], dust_filters=[dust_filter]
        )
        if current_filter_pair not in valid_filter_pairs:
            # TODO: log this instead
            # print(
            #     f"Filter pair OH={oh_filter}, dust={dust_filter} not found in epoch {eid.epoch_id} - skipping lightcurve entry for aperture and vectorial water production."
            # )
            continue

        lightcurve_constructor = partial(
            LightCurveEntry,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
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

        vec_lc_entries_base = [
            lightcurve_constructor(
                q_h2o=q,
                q_h2o_err=q_err,
                dust_redness=r,
                q_source=LightCurveEntrySource.from_vectorial,
            )
            for q, q_err, r in zip(
                vqdf.far_fit_q, vqdf.far_fit_q_err, vqdf.dust_redness_pct_per_hundred_nm
            )
        ]

        vec_lc_entries_fixed = [
            replace(x, rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)))  # type: ignore
            for x in vec_lc_entries_base
        ]

        water_production_lightcurve.extend(vec_lc_entries_fixed)

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
        aperture_hydroxyl_production_column = "q_oh_median"
        aperture_hydroxyl_production_err_column = "q_oh_median_err"
        aperture_dust_redness_column = "dust_redness_pct_per_hundred_nm"

        ap_q_col = apdf[aperture_water_production_column]
        ap_q_err_col = apdf[aperture_water_production_err_column]
        ap_q_oh_col = apdf[aperture_hydroxyl_production_column]
        ap_q_oh_err_col = apdf[aperture_hydroxyl_production_err_column]
        ap_dust_redness_col = apdf[aperture_dust_redness_column]

        ap_lc_entries_base = [
            lightcurve_constructor(
                q_h2o=q,
                q_h2o_err=q_err,
                q_oh=q_oh,
                q_oh_err=q_oh_err,
                dust_redness=r,
                q_source=LightCurveEntrySource.from_aperture,
            )
            for q, q_err, q_oh, q_oh_err, r in zip(
                ap_q_col,
                ap_q_err_col,
                ap_q_oh_col,
                ap_q_oh_err_col,
                ap_dust_redness_col,
            )
        ]

        ap_lc_entries_fixed = [
            replace(x, rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)))  # type: ignore
            for x in ap_lc_entries_base
        ]

        water_production_lightcurve.extend(ap_lc_entries_fixed)

    # print(f"Constructed lightcurve:")
    # print(water_production_lightcurve)
    return water_production_lightcurve


def do_water_production_lightcurve(scp: Products, ref: ProductReference) -> None:

    pkey = ref.key
    assert isinstance(pkey, LightcurveKey)

    print(
        f"Building lightcurve for filter pair OH={pkey.oh_filter} dust={pkey.dust_filter}"
    )

    wp_lc = build_water_production_lightcurve(
        scp=scp,
        oh_filter=pkey.oh_filter,
        dust_filter=pkey.dust_filter,
        stacking_method=pkey.stacking_method,
    )

    scp.save_water_production_lightcurve(water_lc=wp_lc, key=pkey)
    print(
        f"Finished building lightcurve for filter pair OH={pkey.oh_filter} dust={pkey.dust_filter}"
    )


def build_bayesian_water_production_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    dust_redness_sigma: DustReddeningPercent,
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

    current_filter_pair = WaterProductionFilterPair(
        oh_filter=oh_filter, dust_filter=dust_filter
    )
    water_production_lightcurve: LightCurve = []

    for eid in eids:

        # do we have observations in the filter pair for this epoch?
        valid_filter_pairs = get_valid_water_production_filter_pairs(
            eid=eid, oh_filters=[oh_filter], dust_filters=[dust_filter]
        )
        if current_filter_pair not in valid_filter_pairs:
            # TODO: log this instead
            # print(
            #     f"Filter pair OH={oh_filter}, dust={dust_filter} not found in epoch {eid.epoch_id} - skipping lightcurve entry for aperture and vectorial water production."
            # )
            continue

        lightcurve_constructor = partial(
            LightCurveEntry,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            # dust_redness_mean=dust_redness_mean,
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

        vec_lc_entries_base = [
            lightcurve_constructor(
                q_h2o=q,
                q_h2o_err=q_err,
                dust_redness=r,
                q_source=LightCurveEntrySource.from_vectorial,
            )
            for q, q_err, r in zip(
                bqdf.bayes_far_fit_q,
                bqdf.bayes_far_fit_q_err,
                bqdf.dust_redness_pct_per_hundred_nm,
            )
        ]

        vec_lc_entries_fixed = [
            replace(x, rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)))  # type: ignore
            for x in vec_lc_entries_base
        ]

        water_production_lightcurve.extend(vec_lc_entries_fixed)

        aperture_water_production_column = "aperture_matched_q_h2o_median"
        aperture_water_production_err_column = "aperture_matched_q_h2o_median_err"
        aperture_hydroxyl_production_column = "q_oh_median"
        aperture_hydroxyl_production_err_column = "q_oh_median_err"
        aperture_dust_redness_column = "dust_redness_pct_per_hundred_nm"

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
        negative_production_included = (
            lambda x: x[aperture_water_production_column] > -np.inf
        )
        bapdf = bayesian_expectation_of_assembled_fixed_aperture_results(
            df=apdf,
            dust_redness_sigma=dust_redness_sigma,
            constraint_function=negative_production_included,
        )

        ap_q_col = bapdf[aperture_water_production_column]
        ap_q_err_col = bapdf[aperture_water_production_err_column]
        ap_q_oh_col = bapdf[aperture_hydroxyl_production_column]
        ap_q_oh_err_col = bapdf[aperture_hydroxyl_production_err_column]
        ap_dust_redness_col = bapdf[aperture_dust_redness_column]

        ap_lc_entries_base = [
            lightcurve_constructor(
                q_h2o=q,
                q_h2o_err=q_err,
                q_oh=q_oh,
                q_oh_err=q_oh_err,
                dust_redness=r,
                q_source=LightCurveEntrySource.from_aperture,
            )
            for q, q_err, q_oh, q_oh_err, r in zip(
                ap_q_col,
                ap_q_err_col,
                ap_q_oh_col,
                ap_q_oh_err_col,
                ap_dust_redness_col,
            )
        ]

        ap_lc_entries_fixed = [
            replace(x, rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)))  # type: ignore
            for x in ap_lc_entries_base
        ]

        water_production_lightcurve.extend(ap_lc_entries_fixed)

    return water_production_lightcurve


def do_bayesian_water_production_lightcurve(
    scp: Products, ref: ProductReference
) -> None:

    pkey = ref.key
    assert isinstance(pkey, BayesianPriorLightcurveKey)

    blc = build_bayesian_water_production_lightcurve(
        scp=scp,
        oh_filter=pkey.oh_filter,
        dust_filter=pkey.dust_filter,
        stacking_method=pkey.stacking_method,
        dust_redness_sigma=pkey.dust_redness_sigma_pct_per_hundred_nm,
    )

    scp.save_bayesian_water_production_lightcurve(water_lc=blc, key=pkey)

    print(f"Finished building {pkey}!")
