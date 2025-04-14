import pandas as pd
import astropy.units as u

from swift_comet_pipeline.modeling.vectorial_model import water_vectorial_model
from swift_comet_pipeline.modeling.vectorial_model_fit import vectorial_fit
from swift_comet_pipeline.observationlog.epoch_typing import EpochID
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.pipeline_utils.comet_column_density import (
    get_comet_column_density_from_extracted_profile,
)
from swift_comet_pipeline.pipeline_utils.epoch_summary import get_epoch_summary
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent
from swift_comet_pipeline.types.stacking_method import StackingMethod


def blue_spot_detect(
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    epoch_id: EpochID,
    dust_rednesses: list[DustReddeningPercent],
    near_far_radius: u.Quantity,
) -> pd.DataFrame | None:

    epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
    if epoch_summary is None:
        return None

    model_Q = 1e29 / u.s  # type: ignore
    helio_r = epoch_summary.rh_au * u.AU  # type: ignore
    vmr = water_vectorial_model(base_q=model_Q, helio_r=helio_r)

    sub_dataframes = []
    for dust_redness in dust_rednesses:
        col_dens = get_comet_column_density_from_extracted_profile(
            scp=scp,
            epoch_id=epoch_id,
            dust_redness=dust_redness,
            stacking_method=stacking_method,
        )
        far_fit = vectorial_fit(
            comet_column_density=col_dens,
            model_Q=model_Q,
            vmr=vmr,
            r_fit_min=near_far_radius,
            r_fit_max=1.0e10 * u.km,  # type: ignore
        )

        # the column density of the vectorial fit is done on the same radial grid as the column density derived from our data
        excess_cd = col_dens.cd_cm2 - far_fit.vectorial_column_density.cd_cm2

        sub_dataframes.append(
            pd.DataFrame(
                {
                    "rs_km": col_dens.rs_km,
                    "data_coldens": col_dens.cd_cm2,
                    "vec_coldens": far_fit.vectorial_column_density.cd_cm2,
                    "excess_cd": excess_cd,
                    "dust_redness": dust_redness,
                }
            )
        )

    return pd.concat(sub_dataframes)
