from itertools import product
from dataclasses import asdict

from tqdm import tqdm
import pandas as pd
import astropy.units as u

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_pipeline_analysis.column_density_above_background import (
    column_density_above_background,
)
from swift_comet_pipeline.post_pipeline_analysis.epoch_summary import get_epoch_summary
from swift_comet_pipeline.post_pipeline_analysis.vectorial_fitting_reliable import (
    column_density_has_enough_coverage,
    column_density_larger_than_psf_threshold,
    vectorial_fitting_reliable,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


# TODO: move this into a post-lightcurve building step, and make it return a Lightcurve plus columns instead of EpochSummary
def tag_lightcurve_vectorial_fitting_reliability(
    scp: SwiftCometPipeline,
    lc_df: pd.DataFrame,
    stacking_method: StackingMethod,
    dust_rednesses: list[DustReddeningPercent],
    vectorial_fitting_requires: u.Quantity,
) -> pd.DataFrame | None:
    """
    Add columns to the lightcurve dataframe that indicate vectorial fitting suitability for each epoch

    lc_df should be a lightcurve converted to a dataframe with lightcurve_to_dataframe()

    Returns a dataframe with an EpochSummary plus additional columns:
        good_coldens_coverage
        good_psf_threshold
        vectorial_fitting_reliable
    """

    df = lc_df.copy()

    epoch_ids = scp.get_epoch_id_list()
    if epoch_ids is None:
        return None
    df_pieces = []
    for eid, d_redness in tqdm(
        product(epoch_ids, dust_rednesses), total=len(epoch_ids) * len(dust_rednesses)
    ):
        cd_bg = column_density_above_background(
            scp=scp,
            epoch_id=eid,
            dust_redness=d_redness,
            stacking_method=stacking_method,
        )
        # This can fail for epochs that have not been analyzed
        if cd_bg is None:
            continue
        ep_summary = get_epoch_summary(scp=scp, epoch_id=eid)
        if ep_summary is None:
            continue
        bg_df = pd.DataFrame.from_records([asdict(cd_bg)])
        ep_df = pd.DataFrame.from_records([asdict(ep_summary)])
        df = pd.concat([ep_df, bg_df], axis=1)
        df["good_coldens_coverage"] = column_density_has_enough_coverage(cd_bg=cd_bg)
        df["good_psf_threshold"] = column_density_larger_than_psf_threshold(cd_bg=cd_bg)
        df["vectorial_fitting_reliable"] = vectorial_fitting_reliable(
            cd_bg=cd_bg, vectorial_fitting_requires=vectorial_fitting_requires
        )
        df_pieces.append(df)
    if len(df_pieces) == 0:
        return None
    total_df = pd.concat(df_pieces, ignore_index=True)
    return total_df
