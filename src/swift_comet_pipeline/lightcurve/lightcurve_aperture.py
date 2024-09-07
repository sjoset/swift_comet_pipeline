import itertools

import numpy as np
from astropy.time import Time
import astropy.units as u

from swift_comet_pipeline.aperture.plateau_serialize import (
    dust_plateau_list_dict_unserialize,
)
from swift_comet_pipeline.lightcurve.lightcurve import LightCurve, LightCurveEntry
from swift_comet_pipeline.pipeline.files.epoch_subpipeline_files import (
    EpochSubpipelineFiles,
)
from swift_comet_pipeline.pipeline.files.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


def lightcurve_from_aperture_plateaus(
    pipeline_files: PipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
) -> LightCurve | None:
    # TODO: document
    if pipeline_files.epoch_subpipelines is None:
        return None

    lightcurve_entries = []
    for epoch_subpipeline_files in pipeline_files.epoch_subpipelines:
        lc_entries = lightcurve_entries_from_aperture_plateaus(
            epoch_subpipeline_files=epoch_subpipeline_files,
            stacking_method=stacking_method,
            t_perihelion=t_perihelion,
        )
        lightcurve_entries.append(lc_entries)

    lightcurve: list[LightCurveEntry] = [x for x in lightcurve_entries if x is not None]

    # flatten list before returning
    return list(itertools.chain.from_iterable(lightcurve))  # type: ignore


def lightcurve_entries_from_aperture_plateaus(
    epoch_subpipeline_files: EpochSubpipelineFiles,
    stacking_method: StackingMethod,
    t_perihelion: Time,
) -> list[LightCurveEntry] | None:
    # TODO: document

    # if epoch_subpipeline_files is None:
    #     return None

    epoch_subpipeline_files.stacked_epoch.read_product_if_not_loaded()
    stacked_epoch = epoch_subpipeline_files.stacked_epoch.data
    if stacked_epoch is None:
        return None

    helio_r = np.mean(stacked_epoch.HELIO) * u.AU  # type: ignore
    observation_time = Time(np.mean(stacked_epoch.MID_TIME))
    time_from_perihelion_days = float((observation_time - t_perihelion).to_value(u.day))  # type: ignore

    q_vs_r_product = epoch_subpipeline_files.qh2o_vs_aperture_radius_analyses[
        stacking_method
    ]

    if not q_vs_r_product.product_path.exists():
        return None

    q_vs_r_product.read_product_if_not_loaded()
    if q_vs_r_product.data is None:
        return None

    df = q_vs_r_product.data

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
