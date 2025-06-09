from dataclasses import dataclass

import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary


# TODO: move this to types/
@dataclass
class PositionAngles:
    dust_tail_pa: u.Quantity
    ion_tail_pa: u.Quantity


def get_position_angles(
    scp: SwiftCometPipeline, epoch_id: EpochID
) -> PositionAngles | None:

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    hor = Horizons(
        id=scp.spc.jpl_horizons_id,
        location="@swift",
        epochs=Time(epoch_summary.observation_time).jd,
    )
    eph = hor.ephemerides()
    e_df = eph.to_pandas()

    dust_tail_pa = e_df.velocityPA[0]
    ion_tail_pa = e_df.sunTargetPA[0]

    return PositionAngles(
        dust_tail_pa=dust_tail_pa * u.degree, ion_tail_pa=ion_tail_pa * u.degree  # type: ignore
    )
