from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.pipeline_utils.get_uw1_and_uvv import (
    get_uw1_and_uvv_extracted_radial_profiles,
)
from swift_comet_pipeline.types.uw1_uvv_pair import uw1uvv_getter
from swift_comet_pipeline.comet.calculate_column_density import (
    calculate_comet_column_density,
)

from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.lightcurve import LightCurve, LightCurveEntry
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.vectorial_model_fit_type import VectorialFitType


def lightcurve_from_vectorial_fits(
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurve | None:

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None

    lightcurve = [
        lightcurve_entry_from_vectorial_fits(
            scp=scp,
            epoch_id=epoch_id,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
            dust_redness=dust_redness,
            fit_type=fit_type,
            near_far_radius=near_far_radius,
        )
        for epoch_id in epoch_ids
    ]

    return lightcurve


def lightcurve_entry_from_vectorial_fits(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    t_perihelion: Time,
    dust_redness: DustReddeningPercent,
    fit_type: VectorialFitType,
    near_far_radius: u.Quantity,
) -> LightCurveEntry | None:
    # TODO: if the extracted profile is not long enough, then there may be no data for far fit, and vectorial_fit crashes.
    # Handle this case better!

    # TODO: use get_epoch_summary and uw1_uvv getter

    es = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    assert es is not None
    helio_r = es.rh_au * u.AU  # type: ignore
    observation_time = Time(es.observation_time)

    profs = get_uw1_and_uvv_extracted_radial_profiles(
        scp=scp, epoch_id=epoch_id, stacking_method=stacking_method
    )
    assert profs is not None
    uw1_profile, uvv_profile = uw1uvv_getter(profs)

    model_Q = 1e29 / u.s  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r)
    if vmr.column_density_interpolation is None:
        print(
            "No column density interpolation returned from vectorial model! This is a bug! Exiting."
        )
        exit(1)

    ccd = calculate_comet_column_density(
        scp=scp,
        epoch_id=epoch_id,
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
        epoch_id=epoch_id,
        observation_time=observation_time,
        time_from_perihelion_days=float(
            (observation_time - t_perihelion).to_value(u.day)  # type: ignore
        ),
        rh_au=helio_r.to_value(u.AU),  # type: ignore
        q=vec_fit.best_fit_Q.to_value(1 / u.s),  # type: ignore
        q_err=vec_fit.best_fit_Q_err.to_value(1 / u.s),  # type: ignore
        dust_redness=dust_redness,
    )
