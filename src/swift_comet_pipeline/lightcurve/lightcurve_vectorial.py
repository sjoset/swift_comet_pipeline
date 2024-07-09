import numpy as np
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)
from swift_comet_pipeline.comet.comet_radial_profile import (
    radial_profile_from_dataframe_product,
)
from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve import LightCurve, LightCurveEntry
from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter, filter_to_file_string


def lightcurve_from_vectorial_fits(
    pipeline_files: PipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurve | None:

    if pipeline_files.epoch_subpipelines is None:
        return None

    lightcurve = []
    for epoch_subpipeline in pipeline_files.epoch_subpipelines:
        lc_entry = lightcurve_entry_from_vectorial_fits(
            epoch_subpipeline_files=epoch_subpipeline,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=fit_type,
            near_far_radius=near_far_radius,
        )
        lightcurve.append(lc_entry)

    return lightcurve


def lightcurve_entry_from_vectorial_fits(
    epoch_subpipeline_files: EpochSubpipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurveEntry | None:

    if not epoch_subpipeline_files.stacked_epoch.product_path.exists():
        return None

    epoch_subpipeline_files.stacked_epoch.read_product_if_not_loaded()
    stacked_epoch = epoch_subpipeline_files.stacked_epoch.data
    if stacked_epoch is None:
        return None

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    observation_time = Time(np.mean(stacked_epoch.MID_TIME))

    for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
        p = epoch_subpipeline_files.extracted_profiles[filter_type, stacking_method]
        p.read_product_if_not_loaded()
        if p.data is None:
            # print(
            #     f"Could not load radial profile of {epoch_subpipeline_files.parent_epoch.product_path.name} for filter {filter_to_file_string(filter_type=filter_type)}"
            # )
            return None

    uw1_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline_files.extracted_profiles[
            SwiftFilter.uw1, stacking_method
        ].data
    )
    uvv_profile = radial_profile_from_dataframe_product(
        df=epoch_subpipeline_files.extracted_profiles[
            SwiftFilter.uvv, stacking_method
        ].data
    )

    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r, model_backend="rust")
    if vmr.column_density_interpolation is None:
        print(
            "No column density interpolation returned from vectorial model! This is a bug! Exiting."
        )
        exit(1)

    ccd = calculate_comet_column_density(
        stacked_epoch=stacked_epoch,
        uw1_profile=uw1_profile,
        uvv_profile=uvv_profile,
        dust_redness=dust_redness,
        r_min=1 * u.km,  # type: ignore
    )

    if fit_type == VectorialFitType.near_fit:
        fit_radius_start = 1 * u.km  # type: ignore
        fit_radius_stop = near_far_radius  # type: ignore
    elif fit_type == VectorialFitType.far_fit:
        fit_radius_start = near_far_radius  # type: ignore
        # TODO: magic number - this should be large enough for all cases but there should be a better way to set upper limit
        fit_radius_stop = 1.0e10 * u.km  # type: ignore
    elif fit_type == VectorialFitType.full_fit:
        fit_radius_start = 1 * u.km  # type: ignore
        fit_radius_stop = 1.0e10 * u.km  # type: ignore

    vec_fit = vectorial_fit(
        comet_column_density=ccd,
        model_Q=model_Q,
        vmr=vmr,
        r_fit_min=fit_radius_start,
        r_fit_max=fit_radius_stop,
    )

    return LightCurveEntry(
        observation_time=observation_time,
        time_from_perihelion_days=float(
            (observation_time - t_perihelion).to_value(u.day)
        ),
        rh_au=helio_r.to_value(u.AU),  # type: ignore
        q=vec_fit.best_fit_Q.to_value(1 / u.s),  # type: ignore
        q_err=vec_fit.best_fit_Q_err.to_value(1 / u.s),  # type: ignore
        dust_redness=dust_redness,
    )
