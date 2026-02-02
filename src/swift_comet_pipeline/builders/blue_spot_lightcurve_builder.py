from dataclasses import replace
from functools import partial

from swift_comet_pipeline.pipeline.product_system.product_key import (
    BayesianPriorBlueSpotLightcurveKey,
    BlueSpotLightcurveKey,
)
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.modeling.vectorial.molecular_parameters import (
    make_hydroxyl_fragment,
)
from swift_comet_pipeline.post_processing.blue_spot import (
    bayes_blue_spot_df_from_blue_spot_df,
    blue_spot_df_from_vectorial_df,
)
from swift_comet_pipeline.post_processing.vectorial_model_results import (
    assemble_vectorial_model_results,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndex
from swift_comet_pipeline.scp_types.compound.lightcurve import (
    BayesianPriorBlueSpotLightCurve,
    BayesianPriorBlueSpotLightCurveEntry,
    BlueSpotLightCurve,
    BlueSpotLightCurveEntry,
    LightCurveEntrySource,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products


# TODO: move this somewhere else
blue_spot_analysis_max_rh = 3.5 * u.au


# non-bayesian version
def build_blue_spot_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    blue_spot_extent_km: float,
    epochs_to_include: EpochIndex | None = None,
) -> BlueSpotLightCurve:
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

    blue_spot_lightcurve: BlueSpotLightCurve = []

    for eid in eids:

        # if eid.rh_au > 3.0:
        if eid.rh_au > blue_spot_analysis_max_rh.to_value(u.au):
            print(
                f"Skipping {eid.epoch_id}: rh of {eid.rh_au} is too large for blue spot analysis."
            )
            continue

        oh_fragment = make_hydroxyl_fragment()
        scaled_oh_lifetime = oh_fragment.tau_T_s * eid.rh_au**2

        lightcurve_constructor = partial(
            BlueSpotLightCurveEntry,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            blue_spot_extent_km=blue_spot_extent_km,
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

        bs_lc_entries_base = [
            lightcurve_constructor(
                q_oh=q_oh,
                q_oh_err=0,
                q_source=LightCurveEntrySource.from_blue_spot,
                dust_redness=r,
            )
            for r, q_oh in zip(
                bsdf.dust_redness_pct_per_hundred_nm, bsdf.blue_spot_q_oh
            )
        ]
        # apply sign to rh_au for pre-perihelion
        bs_lc_entries = [
            replace(
                x,
                rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)),  # type: ignore
            )
            for x in bs_lc_entries_base
        ]

        blue_spot_lightcurve.extend(bs_lc_entries)

    return blue_spot_lightcurve


def do_blue_spot_lightcurve(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, BlueSpotLightcurveKey)

    bs_lc = build_blue_spot_lightcurve(
        scp=scp,
        oh_filter=ref.key.oh_filter,
        dust_filter=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
        blue_spot_extent_km=ref.key.blue_spot_extent_km,
    )
    scp.save_blue_spot_lightcurve(bs_lc=bs_lc, key=ref.key)


def build_bayesian_prior_blue_spot_lightcurve(
    scp: Products,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
    dust_redness_sigma: DustReddeningPercent,
    blue_spot_extent_km: float,
    epochs_to_include: EpochIndex | None = None,
) -> BayesianPriorBlueSpotLightCurve:
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

    blue_spot_lightcurve: BayesianPriorBlueSpotLightCurve = []

    for eid in eids:

        # if eid.rh_au > 3.0:
        if eid.rh_au > blue_spot_analysis_max_rh.to_value(u.au):
            print(
                f"Skipping {eid.epoch_id}: rh of {eid.rh_au} is too large for blue spot analysis."
            )
            continue

        oh_fragment = make_hydroxyl_fragment()
        scaled_oh_lifetime = oh_fragment.tau_T_s * eid.rh_au**2

        lightcurve_constructor = partial(
            BayesianPriorBlueSpotLightCurveEntry,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            blue_spot_extent_km=blue_spot_extent_km,
            dust_redness_sigma_pct_per_hundred_nm=dust_redness_sigma,
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

        bs_lc_entries_base = [
            lightcurve_constructor(
                q_oh=q_oh,
                q_oh_err=0,
                q_source=LightCurveEntrySource.from_blue_spot,
                dust_redness=r,
            )
            for r, q_oh in zip(
                bayes_bsdf.dust_redness_pct_per_hundred_nm,
                bayes_bsdf.bayes_blue_spot_q_oh,
            )
        ]
        # apply sign to rh_au for pre-perihelion
        bs_lc_entries = [
            replace(
                x,
                rh_au=x.rh_au * np.sign(x.time_from_perihelion.to_value(u.day)),  # type: ignore
            )
            for x in bs_lc_entries_base
        ]

        blue_spot_lightcurve.extend(bs_lc_entries)

    return blue_spot_lightcurve


def do_bayesian_prior_blue_spot_lightcurve(
    scp: Products, ref: ProductReference
) -> None:

    assert isinstance(ref.key, BayesianPriorBlueSpotLightcurveKey)

    bpbs_lc = build_bayesian_prior_blue_spot_lightcurve(
        scp=scp,
        oh_filter=ref.key.oh_filter,
        dust_filter=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
        blue_spot_extent_km=ref.key.blue_spot_extent_km,
        dust_redness_sigma=ref.key.dust_redness_sigma_pct_per_hundred_nm,
    )
    scp.save_bayesian_prior_blue_spot_lightcurve(bs_lc=bpbs_lc, key=ref.key)
