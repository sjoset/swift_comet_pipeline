import numpy as np
import pandas as pd
import astropy.units as u

from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.epoch_summary import get_epoch_summary


def tag_lightcurve_with_epoch_id(
    scp: SwiftCometPipeline, lc_df: pd.DataFrame
) -> pd.DataFrame | None:
    """
    lc_df should be a lightcurve converted to a dataframe with lightcurve_to_dataframe()

    NOTE: This function rounds the time to the nearest day, so if there are two epochs very close to one another,
    this may cause an issue.  A warning is printed in this case, but the dataframe is returned with no errors.
    """

    epoch_ids = scp.get_epoch_id_list()
    if epoch_ids is None:
        return None

    df = lc_df.copy()
    df["epoch_id"] = pd.NA

    for epoch_id in epoch_ids:

        # compare the epoch's time from perihelion to the lightcurve time from perihelion: if they match, tag it with this epoch_id
        epoch_summary = get_epoch_summary(scp=scp, epoch_id=epoch_id)
        if epoch_summary is None:
            return None
        epoch_summary_tp = np.round(epoch_summary.time_from_perihelion.to_value(u.day))  # type: ignore

        mask = np.round(df.time_from_perihelion_days) == epoch_summary_tp

        if df.loc[mask, "epoch_id"].notna().any():
            print(f"Warning: epoch id {epoch_id} overwriting results!")

        df.loc[mask, "epoch_id"] = epoch_id

    return df
