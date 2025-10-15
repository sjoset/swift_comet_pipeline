from functools import partial

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from swift_comet_pipeline.modeling.water_production.fluorescence_OH import (
    oh_flux_to_num_oh,
)
from swift_comet_pipeline.modeling.water_production.flux_OH import (
    oh_flux_from_oh_count_rate,
)
from swift_comet_pipeline.modeling.water_production.num_OH_to_Q import (
    num_oh_within_r_to_q_h2o_vectorial,
)
from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductReference,
    Products,
    WaterProductionKey,
)
from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive.dust_reddening_percent import (
    DustReddeningPercent,
)


# def get_production_plateaus(
#     sorted_q_vs_r: list[QvsApertureRadiusEntry],
# ) -> ReddeningToProductionPlateauListDict:
#     """
#     sorted_q_vs_r should have been previously sorted by dust_redness to make sure the entries are contiguous so groupby() catches all of them
#     """
#
#     by_redness = lambda x: x.dust_redness
#
#     q_plateau_list_dict = {}
#     for dust_redness, q_vs_aperture_radius_entry_at_redness in groupby(
#         sorted_q_vs_r, key=by_redness
#     ):
#         qvarear = list(q_vs_aperture_radius_entry_at_redness)
#
#         q_plateau_list = find_production_plateaus(q_vs_aperture_radius_list=qvarear)
#         q_plateau_list_dict[dust_redness] = q_plateau_list
#
#     return q_plateau_list_dict
#
#
# def get_production_plateaus_from_yaml(
#     yaml_dict: dict,
# ) -> ReddeningToProductionPlateauListDict:
#     """
#     Takes a ReddeningToProductionPlateauListDict that was stored as a dict in metadata and reconstructs into ReddeningToProductionPlateauListDict
#     """
#
#     q_plateau_list_dict: ReddeningToProductionPlateauListDict = {}
#
#     for dust_redness, plateau_list_dict in yaml_dict.items():
#         q_plateau_list_dict[dust_redness] = [
#             dict_to_production_plateau(x) for x in plateau_list_dict
#         ]
#
#     return q_plateau_list_dict


# TODO: range of rednesses and combine results with pd.concat
def do_aperture_water_production(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, WaterProductionKey)

    # TODO: Jorda 2008 empirical water production rates for V-band

    oh_subpipe_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.oh_filter,
        stacking_method=ref.key.stacking_method,
    )
    dust_subpipe_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.dust_filter,
        stacking_method=ref.key.stacking_method,
    )
    assert oh_subpipe_key.epoch_id == dust_subpipe_key.epoch_id

    oh_aa = scp.load_annular_aperture_analysis(key=oh_subpipe_key)
    dust_aa = scp.load_annular_aperture_analysis(key=dust_subpipe_key)

    assert np.sum(oh_aa.aperture_r_km - dust_aa.aperture_r_km) == 0.0

    dust_redness = DustReddeningPercent(40.0)
    beta = beta_parameter(dust_redness=dust_redness)
    water_df = oh_aa[
        [
            "aperture_r_km",
            "aperture_r_pix",
            "cumulative_sum",
            "cumulative_sum_variance",
            "cumulative_median_signal",
            "cumulative_median_variance",
            "cumulative_sum_magnitude",
            "cumulative_sum_magnitude_variance",
        ]
    ].copy()

    # TODO: generalize this into 'source oh column' and 'destination column name template' to get sum and median
    water_df["oh_counts_sum"] = water_df.cumulative_sum - beta * dust_aa.cumulative_sum
    water_df["oh_counts_sum_variance"] = (
        water_df.cumulative_sum_variance + beta**2 * dust_aa.cumulative_sum_variance
    )
    water_df["oh_counts_sum_err"] = np.sqrt(water_df.oh_counts_sum_variance)

    water_df["oh_counts_median"] = (
        water_df.cumulative_median_signal - beta * dust_aa.cumulative_median_signal
    )
    water_df["oh_counts_median_variance"] = (
        water_df.cumulative_median_variance
        + beta**2 * dust_aa.cumulative_median_variance
    )
    water_df["oh_counts_median_err"] = np.sqrt(water_df.oh_counts_median_variance)

    # oh flux
    water_df["oh_count_rate_sum"] = [
        CountRate(value=x, sigma=s)
        for x, s in zip(water_df.oh_counts_sum, water_df.oh_counts_sum_err)
    ]
    water_df["oh_flux_sum_val"] = water_df.oh_count_rate_sum.apply(
        oh_flux_from_oh_count_rate
    )
    water_df["oh_flux_sum"] = water_df.oh_flux_sum_val.apply(lambda x: x.value)
    water_df["oh_flux_sum_variance"] = water_df.oh_flux_sum_val.apply(
        lambda x: x.sigma**2
    )
    water_df["oh_flux_sum_err"] = water_df.oh_flux_sum_val.apply(lambda x: x.sigma)

    eid = scp.load_epoch_index_entry(epoch_id=oh_subpipe_key.epoch_id)
    assert eid is not None
    flux_to_num_func = partial(
        oh_flux_to_num_oh,
        rh_au=eid.rh_au,
        helio_v_kms=eid.helio_v_kms,
        delta_au=eid.delta_au,
    )
    water_df["num_oh_sum_val"] = water_df.oh_flux_sum_val.apply(flux_to_num_func)
    water_df["num_oh_sum"] = water_df.num_oh_sum_val.apply(lambda x: x.value)
    water_df["num_oh_sum_var"] = water_df.num_oh_sum_val.apply(lambda x: x.sigma**2)
    water_df["num_oh_sum_err"] = water_df.num_oh_sum_val.apply(lambda x: x.sigma)

    q_vals = [
        num_oh_within_r_to_q_h2o_vectorial(
            rh_au=eid.rh_au, num_oh=n, within_r=r_km * u.km  # type: ignore
        )
        for n, r_km in zip(water_df.num_oh_sum_val, water_df.aperture_r_km)
    ]

    # TODO: add the 'equivalent' vectorial production rates
    water_df["q_h2o_sum_val"] = q_vals
    water_df["q_h2o_sum"] = water_df.q_h2o_sum_val.apply(lambda x: x.value)
    water_df["q_h2o_sum_var"] = water_df.q_h2o_sum_val.apply(lambda x: x.sigma**2)
    water_df["q_h2o_sum_err"] = water_df.q_h2o_sum_val.apply(lambda x: x.sigma)

    # TODO: plateau detection and add to metadata

    # TODO: save data: make dataclass to describe rows and convert to dataclass before handing off to scp

    # TODO: move plotting to test functions

    for i in range(4):
        plt.fill_between(
            water_df.aperture_r_km,
            water_df.q_h2o_sum + i * water_df.q_h2o_sum_err,
            water_df.q_h2o_sum - i * water_df.q_h2o_sum_err,
            alpha=0.2,
            color="#688894",
        )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.cumulative_sum_magnitude
    #         + i * water_df.cumulative_sum_magnitude_variance,
    #         water_df.cumulative_sum_magnitude
    #         - i * water_df.cumulative_sum_magnitude_variance,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.num_oh_sum + i * water_df.num_oh_sum_err,
    #         water_df.num_oh_sum - i * water_df.num_oh_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_flux_sum + i * water_df.oh_flux_sum_err,
    #         water_df.oh_flux_sum - i * water_df.oh_flux_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )

    # for i in range(4):
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_counts_sum + i * water_df.oh_counts_sum_err,
    #         water_df.oh_counts_sum - i * water_df.oh_counts_sum_err,
    #         alpha=0.2,
    #         color="#688894",
    #     )
    #     plt.fill_between(
    #         water_df.aperture_r_km,
    #         water_df.oh_counts_median + i * water_df.oh_counts_median_err,
    #         water_df.oh_counts_median - i * water_df.oh_counts_median_err,
    #         alpha=0.2,
    #         color="#afac7c",
    #     )

    plt.yscale("log")
    plt.xscale("log")

    # plt.xlim(0, 500000)
    plt.ylim(
        0,
    )
    plt.show()

    print("done")
