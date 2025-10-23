from dataclasses import fields
from functools import partial

import pandas as pd
import astropy.units as u

from swift_comet_pipeline.modeling.magnitude import magnitude_from_count_rate
from swift_comet_pipeline.photometry.aperture.aperture_count_rate import (
    count_rate_variance_in_aperture,
)
from swift_comet_pipeline.photometry.dust.afrho import calculate_afrho_in_cm
from swift_comet_pipeline.photometry.dust.halley_marcus import (
    halley_marcus_curve_interpolation,
)
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductReference,
    Products,
)
from swift_comet_pipeline.scp_types.compound.count_rate import CountRate
from swift_comet_pipeline.scp_types.compound.magnitude import Magnitude
from swift_comet_pipeline.scp_types.primitive.afrho_from_aperture_photometry import (
    afrho_aperture_photometry_analysis_from_dataframe,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_profile import (
    afrho_from_radial_profile_from_dataframe,
)
from swift_comet_pipeline.scp_types.primitive.annular_aperture_photometry_analysis import (
    dataframe_from_annular_aperture_photometry_analysis,
)
from swift_comet_pipeline.scp_types.primitive import *


def do_afrho_from_aperture_photometry_analysis(
    scp: Products, ref: ProductReference
) -> None:

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    eid = scp.load_epoch_index_entry(epoch_id=pkey.epoch_id)
    assert eid is not None

    afrho_phase_correction_func = halley_marcus_curve_interpolation()
    assert afrho_phase_correction_func is not None
    afrho_phase_correction = afrho_phase_correction_func(eid.phase_angle_deg)

    aapa_load_res = scp.load_annular_aperture_analysis(key=pkey)
    assert aapa_load_res is not None
    aapa, _ = aapa_load_res
    aapa_df = dataframe_from_annular_aperture_photometry_analysis(aapa=aapa)

    photometry_columns = list(aapa_df.columns)
    columns_to_copy = [
        x.name
        for x in fields(AfrhoFromAperturePhotometryAnalysisEntry)
        if x.name in photometry_columns
    ]
    afrho_df = aapa_df[columns_to_copy]

    # set up Magnitudes for error propogation
    afrho_df["cumulative_median_magnitude_val"] = afrho_df.apply(
        lambda row: Magnitude(
            value=row.cumulative_median_magnitude,
            sigma=row.cumulative_median_magnitude_err,
        ),
        axis=1,
    )
    afrho_df["cumulative_sum_magnitude_val"] = afrho_df.apply(
        lambda row: Magnitude(
            value=row.cumulative_sum_magnitude,
            sigma=row.cumulative_sum_magnitude_err,
        ),
        axis=1,
    )

    afrho_func = partial(
        calculate_afrho_in_cm,
        delta=eid.delta_au * u.AU,  # type: ignore
        rh=eid.rh_au * u.AU,  # type: ignore
        filter_type=pkey.filter_type,
    )

    # afrho from median apertures
    afrho_df["cumulative_afrho_median_cm_val"] = afrho_df.apply(
        lambda row: afrho_func(
            rho=row.aperture_r_km * u.km, mag_in_filter=row.cumulative_median_magnitude_val  # type: ignore
        ),
        axis=1,
    )
    afrho_df["cumulative_afrho_median_cm"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_median_cm_val.value,
        axis=1,
    )
    afrho_df["cumulative_afrho_median_cm_err"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_median_cm_val.sigma,
        axis=1,
    )
    afrho_df["cumulative_afrho_median_cm_var"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_median_cm_val.sigma**2,
        axis=1,
    )

    # afrho from aperture sums
    afrho_df["cumulative_afrho_sum_cm_val"] = afrho_df.apply(
        lambda row: afrho_func(
            rho=row.aperture_r_km * u.km, mag_in_filter=row.cumulative_sum_magnitude_val  # type: ignore
        ),
        axis=1,
    )
    afrho_df["cumulative_afrho_sum_cm"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_sum_cm_val.value, axis=1
    )
    afrho_df["cumulative_afrho_sum_cm_err"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_sum_cm_val.sigma, axis=1
    )
    afrho_df["cumulative_afrho_sum_cm_var"] = afrho_df.apply(
        lambda row: row.cumulative_afrho_sum_cm_val.sigma**2, axis=1
    )

    afrho_df["cumulative_afrho_zero_median_cm"] = (
        afrho_df.cumulative_afrho_median_cm / afrho_phase_correction
    )
    afrho_df["cumulative_afrho_zero_median_cm_err"] = (
        afrho_df.cumulative_afrho_median_cm_err / afrho_phase_correction
    )
    afrho_df["cumulative_afrho_zero_median_cm_var"] = (
        afrho_df.cumulative_afrho_median_cm_var / afrho_phase_correction
    )

    afrho_df["cumulative_afrho_zero_sum_cm"] = (
        afrho_df.cumulative_afrho_sum_cm / afrho_phase_correction
    )
    afrho_df["cumulative_afrho_zero_sum_cm_err"] = (
        afrho_df.cumulative_afrho_sum_cm_err / afrho_phase_correction
    )
    afrho_df["cumulative_afrho_zero_sum_cm_var"] = (
        afrho_df.cumulative_afrho_sum_cm_var / afrho_phase_correction
    )

    assert isinstance(afrho_df, pd.DataFrame)
    afapa = afrho_aperture_photometry_analysis_from_dataframe(df=afrho_df)

    scp.save_afrho_from_aperture_photometry(afapa=afapa, key=pkey)


def do_afrho_from_radial_profile(scp: Products, ref: ProductReference) -> None:

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    # epoch info
    eid = scp.load_epoch_index_entry(epoch_id=pkey.epoch_id)
    assert eid is not None
    exposure_time_s = eid.exposure_times[pkey.filter_type]

    # afrho phase correction
    afrho_phase_correction_func = halley_marcus_curve_interpolation()
    assert afrho_phase_correction_func is not None
    afrho_phase_correction = afrho_phase_correction_func(eid.phase_angle_deg)

    # background
    bg_result = scp.load_background_result(key=pkey)
    assert bg_result is not None

    # profile
    rad_prof = scp.load_extracted_radial_profile(key=pkey)

    df = pd.DataFrame(
        {"r_pix": rad_prof.profile_axis_rs, "count_rate": rad_prof.pixel_values}
    )
    df["r_km"] = df.r_pix * eid.km_per_pix

    # we have count rate as a function of radius - turn that into totals count within radius

    df["inner_r_pix"] = df.r_pix
    df["outer_r_pix"] = df.r_pix.shift(-1)
    df["outer_r_pix"] = df.outer_r_pix.fillna(df.r_pix)
    df["annulus_area_pix"] = np.pi * (df.outer_r_pix**2 - df.inner_r_pix**2)

    df["annulus_count_rate"] = df.annulus_area_pix * df.count_rate
    df["annulus_count_rate_var"] = df.apply(
        lambda row: count_rate_variance_in_aperture(
            total_count_rate=row.annulus_count_rate, exposure_time_s=exposure_time_s
        ),
        axis=1,
    )
    df["annulus_count_rate_err"] = np.sqrt(df.annulus_count_rate_var)

    df["cumulative_count_rate"] = df.annulus_count_rate.cumsum()
    df["cumulative_count_rate_var"] = df.annulus_count_rate_var.cumsum()
    df["cumulative_count_rate_err"] = np.sqrt(df.cumulative_count_rate_var)
    df["cumulative_count_rate_val"] = df.apply(
        lambda row: CountRate(
            value=row.cumulative_count_rate, sigma=row.cumulative_count_rate_err
        ),
        axis=1,
    )

    magnitude_func = partial(magnitude_from_count_rate, filter_type=pkey.filter_type)
    df["cumulative_magnitude_val"] = df.apply(
        lambda row: magnitude_func(count_rate=row.cumulative_count_rate_val), axis=1
    )
    df["cumulative_magnitude"] = df.apply(
        lambda row: row.cumulative_magnitude_val.value, axis=1
    )
    df["cumulative_magnitude_err"] = df.apply(
        lambda row: row.cumulative_magnitude_val.sigma, axis=1
    )
    df["cumulative_magnitude_var"] = df.cumulative_magnitude_err**2

    afrho_func = partial(
        calculate_afrho_in_cm,
        delta=eid.delta_au * u.AU,  # type: ignore
        rh=eid.rh_au * u.AU,  # type: ignore
        filter_type=pkey.filter_type,
    )

    df["cumulative_afrho_cm_val"] = df.apply(
        lambda row: afrho_func(
            rho=row.r_km * u.km, mag_in_filter=row.cumulative_magnitude_val  # type: ignore
        ),
        axis=1,
    )
    df["cumulative_afrho_cm"] = df.apply(
        lambda row: row.cumulative_afrho_cm_val.value, axis=1
    )
    df["cumulative_afrho_cm_err"] = df.apply(
        lambda row: row.cumulative_afrho_cm_val.sigma, axis=1
    )
    df["cumulative_afrho_cm_var"] = np.pow(df.cumulative_afrho_cm_err, 2)

    df["cumulative_afrho_zero_cm"] = df.cumulative_afrho_cm / afrho_phase_correction
    df["cumulative_afrho_zero_cm_var"] = (
        df.cumulative_afrho_cm_var / afrho_phase_correction
    )
    df["cumulative_afrho_zero_cm_err"] = (
        df.cumulative_afrho_cm_err / afrho_phase_correction
    )

    # TODO: some of the pixel radii are 0 leading to a division error - remove those entries before calculating

    # TODO: trim the last row off of the dataframe: last annulus is invalid
    # TODO: the first annulus uses first radius as the inner r - we miss the central nucleus pixel - is this okay?
    afrp = afrho_from_radial_profile_from_dataframe(df=df)

    scp.save_afrho_from_profile(afrp=afrp, key=pkey)
