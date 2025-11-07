import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from swift_comet_pipeline.scp_types.primitive import *


def get_position_angles(
    jpl_horizons_id: str,
    at_time: pd.Timestamp,
) -> TailPositionAngles:

    hor = Horizons(
        id=jpl_horizons_id,
        location="@swift",
        epochs=Time(at_time).jd,
    )
    eph = hor.ephemerides()  # type: ignore
    e_df = eph.to_pandas()

    dust_tail_pa = e_df.velocityPA[0]
    ion_tail_pa = e_df.sunTargetPA[0]

    return TailPositionAngles(
        dust_tail_pa=dust_tail_pa * u.degree, ion_tail_pa=ion_tail_pa * u.degree  # type: ignore
    )
