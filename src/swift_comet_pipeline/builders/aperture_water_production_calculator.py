from functools import partial

import numpy as np
import pandas as pd
import astropy.units as u

from swift_comet_pipeline.modeling.water_production.fluorescence_OH import (
    oh_flux_to_num_oh,
)
from swift_comet_pipeline.modeling.water_production.flux_OH import (
    oh_flux_from_oh_count_rate,
)
from swift_comet_pipeline.modeling.water_production.num_OH_to_Q import (
    num_oh_to_q_h2o_vectorial,
    num_oh_within_r_to_q_h2o_vectorial,
)
from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.photometry.dust.reddening_translate import (
    recalculate_redness_with_new_filter_pair,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductReference,
    Products,
    ContinuumSubtractionKey,
)
from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.primitive.annular_aperture_photometry_analysis import (
    dataframe_from_annular_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive.aperture_water_production_analysis import (
    aperture_water_production_analysis_from_dataframe,
)
from swift_comet_pipeline.swift.filters.filter_wavelengths import (
    calculate_mid_wavelength_nm,
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


def do_aperture_water_production(scp: Products, ref: ProductReference) -> None:
    assert isinstance(ref.key, ContinuumSubtractionKey)

    # gather the photometry results from the filters we are using as dust/oh
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

    oh_load_result = scp.load_annular_aperture_analysis(key=oh_subpipe_key)
    dust_load_result = scp.load_annular_aperture_analysis(key=dust_subpipe_key)
    assert oh_load_result is not None and dust_load_result is not None

    oh_aapa, oh_metadata = oh_load_result
    dust_aapa, dust_metadata = dust_load_result

    oh_aapa_df = dataframe_from_annular_aperture_photometry_analysis(aapa=oh_aapa)
    dust_aapa_df = dataframe_from_annular_aperture_photometry_analysis(aapa=dust_aapa)

    # The product key holds the redness at a certain reference mid wavelength - convert the redness to the proper value
    # for the filters in use while we do the calculations
    # All of the products are stored on disk referring to the dust redness relative to redness_mid_wavelength_nm in the config file
    untransformed_redness = ref.key.dust_redness_pct_per_hundred_nm
    from_mid_wavelength = scp.cfg.redness_mid_wavelength_nm
    to_mid_wavelength = calculate_mid_wavelength_nm(
        filter_one=oh_subpipe_key.filter_type, filter_two=dust_subpipe_key.filter_type
    )
    transformed_dust_redness = np.round(
        recalculate_redness_with_new_filter_pair(
            known_redness=untransformed_redness,
            old_mid_wave_nm=from_mid_wavelength,
            new_filter_one=oh_subpipe_key.filter_type,
            new_filter_two=dust_subpipe_key.filter_type,
        )
    )
    print(
        f"Epoch ID: {ref.key.epoch_id}\t{ref.key.oh_filter} || {ref.key.dust_filter} || {ref.key.stacking_method}"
    )
    print(
        f"Transforming given redness of {untransformed_redness} at mid wave {from_mid_wavelength} to {transformed_dust_redness} at mid wave {to_mid_wavelength}."
    )

    # check that the annular apertures are all the same size
    # TODO: rewrite using an aperture metadata dataclass
    assert (
        oh_metadata["max_aperture_radius_km"] == dust_metadata["max_aperture_radius_km"]
    )
    assert (
        oh_metadata["num_concentric_apertures"]
        == dust_metadata["num_concentric_apertures"]
    )
    assert np.sum(oh_aapa_df.aperture_r_km - dust_aapa_df.aperture_r_km) == 0.0

    beta = beta_parameter(
        # dust_redness=ref.key.dust_redness_pct_per_hundred_nm_,
        dust_redness=transformed_dust_redness,
        oh_filter=ref.key.oh_filter,
        dust_filter=ref.key.dust_filter,
    )
    water_df = oh_aapa_df[
        [
            "aperture_r_km",
            "aperture_r_pix",
            "aperture_dr_pix",
            "aperture_dr_km",
        ]
    ].copy()
    assert isinstance(water_df, pd.DataFrame)

    # TODO: rewrite for _DataframeColumnAndErrorSet
    water_df["oh_counts_sum"] = (
        oh_aapa_df.cumulative_sum - beta * dust_aapa_df.cumulative_sum
    )
    water_df["oh_counts_sum_variance"] = (
        oh_aapa_df.cumulative_sum_variance
        + beta**2 * dust_aapa_df.cumulative_sum_variance
    )
    water_df["oh_counts_sum_err"] = np.sqrt(water_df.oh_counts_sum_variance)

    water_df["oh_counts_median"] = (
        oh_aapa_df.cumulative_area_scaled_median
        - beta * dust_aapa_df.cumulative_area_scaled_median
    )
    water_df["oh_counts_median_variance"] = (
        oh_aapa_df.cumulative_area_scaled_median_variance
        + beta**2 * dust_aapa_df.cumulative_area_scaled_median_variance
    )
    water_df["oh_counts_median_err"] = np.sqrt(water_df.oh_counts_median_variance)

    oh_flux_func = partial(
        oh_flux_from_oh_count_rate, filter_type=oh_subpipe_key.filter_type
    )
    # oh flux
    water_df["oh_count_rate_sum"] = [
        CountRate(value=x, sigma=s)
        for x, s in zip(water_df.oh_counts_sum, water_df.oh_counts_sum_err)
    ]
    water_df["oh_flux_sum_val"] = water_df.oh_count_rate_sum.apply(oh_flux_func)
    water_df["oh_flux_sum"] = water_df.oh_flux_sum_val.apply(lambda x: x.value)
    water_df["oh_flux_sum_variance"] = water_df.oh_flux_sum_val.apply(
        lambda x: x.sigma**2
    )
    water_df["oh_flux_sum_err"] = water_df.oh_flux_sum_val.apply(lambda x: x.sigma)

    water_df["oh_count_rate_median"] = [
        CountRate(value=x, sigma=s)
        for x, s in zip(water_df.oh_counts_median, water_df.oh_counts_median_err)
    ]
    water_df["oh_flux_median_val"] = water_df.oh_count_rate_median.apply(oh_flux_func)
    water_df["oh_flux_median"] = water_df.oh_flux_median_val.apply(lambda x: x.value)
    water_df["oh_flux_median_variance"] = water_df.oh_flux_median_val.apply(
        lambda x: x.sigma**2
    )
    water_df["oh_flux_median_err"] = water_df.oh_flux_median_val.apply(
        lambda x: x.sigma
    )

    # oh number
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
    water_df["num_oh_sum_variance"] = water_df.num_oh_sum_val.apply(
        lambda x: x.sigma**2
    )
    water_df["num_oh_sum_err"] = water_df.num_oh_sum_val.apply(lambda x: x.sigma)

    water_df["num_oh_median_val"] = water_df.oh_flux_median_val.apply(flux_to_num_func)
    water_df["num_oh_median"] = water_df.num_oh_median_val.apply(lambda x: x.value)
    water_df["num_oh_median_variance"] = water_df.num_oh_median_val.apply(
        lambda x: x.sigma**2
    )
    water_df["num_oh_median_err"] = water_df.num_oh_median_val.apply(lambda x: x.sigma)

    # aperture-matched production rates
    aperture_matched_q_vals_sum = [
        num_oh_within_r_to_q_h2o_vectorial(
            rh_au=eid.rh_au, num_oh=n, within_r=r_km * u.km  # type: ignore
        )
        for n, r_km in zip(water_df.num_oh_sum_val, water_df.aperture_r_km)
    ]
    water_df["aperture_matched_q_h2o_sum_val"] = aperture_matched_q_vals_sum
    water_df["aperture_matched_q_h2o_sum"] = (
        water_df.aperture_matched_q_h2o_sum_val.apply(lambda x: x.value)
    )
    water_df["aperture_matched_q_h2o_sum_variance"] = (
        water_df.aperture_matched_q_h2o_sum_val.apply(lambda x: x.sigma**2)
    )
    water_df["aperture_matched_q_h2o_sum_err"] = (
        water_df.aperture_matched_q_h2o_sum_val.apply(lambda x: x.sigma)
    )

    aperture_matched_q_vals_median = [
        num_oh_within_r_to_q_h2o_vectorial(
            rh_au=eid.rh_au, num_oh=n, within_r=r_km * u.km  # type: ignore
        )
        for n, r_km in zip(water_df.num_oh_median_val, water_df.aperture_r_km)
    ]
    water_df["aperture_matched_q_h2o_median_val"] = aperture_matched_q_vals_median
    water_df["aperture_matched_q_h2o_median"] = (
        water_df.aperture_matched_q_h2o_median_val.apply(lambda x: x.value)
    )
    water_df["aperture_matched_q_h2o_median_variance"] = (
        water_df.aperture_matched_q_h2o_median_val.apply(lambda x: x.sigma**2)
    )
    water_df["aperture_matched_q_h2o_median_err"] = (
        water_df.aperture_matched_q_h2o_median_val.apply(lambda x: x.sigma)
    )

    # oh-equivalent production rates
    equivalent_q_vals_sum = [
        num_oh_to_q_h2o_vectorial(rh_au=eid.rh_au, num_oh=n)
        for n in water_df.num_oh_sum_val
    ]
    equivalent_q_vals_median = [
        num_oh_to_q_h2o_vectorial(rh_au=eid.rh_au, num_oh=n)
        for n in water_df.num_oh_median_val
    ]
    water_df["equivalent_q_h2o_sum_val"] = equivalent_q_vals_sum
    water_df["equivalent_q_h2o_sum"] = water_df.equivalent_q_h2o_sum_val.apply(
        lambda x: x.value
    )
    water_df["equivalent_q_h2o_sum_variance"] = water_df.equivalent_q_h2o_sum_val.apply(
        lambda x: x.sigma**2
    )
    water_df["equivalent_q_h2o_sum_err"] = water_df.equivalent_q_h2o_sum_val.apply(
        lambda x: x.sigma
    )

    water_df["equivalent_q_h2o_median_val"] = equivalent_q_vals_median
    water_df["equivalent_q_h2o_median"] = water_df.equivalent_q_h2o_median_val.apply(
        lambda x: x.value
    )
    water_df["equivalent_q_h2o_median_variance"] = (
        water_df.equivalent_q_h2o_median_val.apply(lambda x: x.sigma**2)
    )
    water_df["equivalent_q_h2o_median_err"] = (
        water_df.equivalent_q_h2o_median_val.apply(lambda x: x.sigma)
    )

    # TODO: plateau detection and add to metadata

    water_df["dust_redness_pct_per_hundred_nm"] = (
        ref.key.dust_redness_pct_per_hundred_nm
    )
    awpa = aperture_water_production_analysis_from_dataframe(df=water_df)

    scp.save_aperture_water_production_analysis(awpa=awpa, key=ref.key)
