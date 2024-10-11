import itertools

import numpy as np
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.aperture.plateau_serialize import (
    dust_plateau_list_dict_unserialize,
)
from swift_comet_pipeline.lightcurve.lightcurve import LightCurve, LightCurveEntry
from swift_comet_pipeline.observationlog.epoch import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


def lightcurve_from_aperture_plateaus(
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    t_perihelion: Time,
) -> LightCurve | None:

    # TODO: document

    epoch_ids = scp.get_epoch_id_list()
    assert epoch_ids is not None

    lightcurve_entries = []
    for epoch_id in epoch_ids:
        lc_entries = lightcurve_entries_from_aperture_plateaus(
            scp=scp,
            epoch_id=epoch_id,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
        )
        lightcurve_entries.append(lc_entries)

    lightcurve: list[LightCurveEntry] = [x for x in lightcurve_entries if x is not None]

    # flatten list before returning
    return list(itertools.chain.from_iterable(lightcurve))  # type: ignore


def lightcurve_entries_from_aperture_plateaus(
    scp: SwiftCometPipeline,
    epoch_id: EpochID,
    stacking_method: StackingMethod,
    t_perihelion: Time,
) -> list[LightCurveEntry] | None:
    # TODO: document

    stacked_epoch = scp.get_product_data(
        pf=PipelineFilesEnum.epoch_post_stack, epoch_id=epoch_id
    )
    assert stacked_epoch is not None

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    observation_time = Time(np.mean(stacked_epoch.MID_TIME))
    time_from_perihelion_days = float((observation_time - t_perihelion).to_value(u.day))  # type: ignore

    df = scp.get_product_data(
        pf=PipelineFilesEnum.aperture_analysis,
        epoch_id=epoch_id,
        stacking_method=stacking_method,
    )
    assert df is not None

    q_plateau_list_dict = dust_plateau_list_dict_unserialize(df.attrs)

    lc_entries = []
    dust_rednesses = q_plateau_list_dict.keys()
    for dust_redness in dust_rednesses:
        for plateau in q_plateau_list_dict[dust_redness]:
            q = (plateau.begin_q + plateau.end_q) / 2.0
            q_err = abs(plateau.end_q - plateau.begin_q) / 2.0
            lightcurve_entry = LightCurveEntry(
                observation_time=observation_time,
                time_from_perihelion_days=time_from_perihelion_days,
                rh_au=helio_r.to_value(u.AU),  # type: ignore
                q=q,
                q_err=q_err,
                dust_redness=dust_redness,
            )
            lc_entries.append(lightcurve_entry)

    return lc_entries
