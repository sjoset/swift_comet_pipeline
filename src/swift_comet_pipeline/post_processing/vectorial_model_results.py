import pandas as pd


from swift_comet_pipeline.post_processing.add_epoch_index_entry_to_dataframe import (
    add_epoch_index_entry_to_dataframe,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    ContinuumSubtractionKey,
    Products,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import EpochIndexEntry
from swift_comet_pipeline.scp_types.compound.radial_profile_water_production import (
    RadialProfileWaterProductionAnalysis,
    json_from_radial_profile_water_production_analysis,
)


def dataframe_from_radial_water_production_analysis(
    rpwpa: RadialProfileWaterProductionAnalysis | None,
    dust_redness: DustReddeningPercent,
) -> pd.DataFrame | None:
    """
    Transforms a RadialProfileWaterProductionAnalysis into a dataframe with columns holding the
    near, far, full fits along with the measured oh column density, all at the given redness
    """

    if rpwpa is None:
        return None

    # df = pd.DataFrame({"oh_column_density": rpwpa.oh_column_density}, index=[0])
    df = pd.DataFrame(
        {"oh_column_density": rpwpa.oh_column_density}, index=pd.Series([0])
    )
    df["dust_redness_pct_per_hundred_nm"] = dust_redness

    for ft in VectorialFitType.all_types():
        df[ft] = getattr(rpwpa, ft)

    return df


def assemble_vectorial_model_results(
    scp: Products,
    eid: EpochIndexEntry,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
    stacking_method: StackingMethod,
) -> pd.DataFrame:
    """
    Gather Q(h2o) for the the given epoch and filters derived from vectorial model fitting of the OH column density at every redness:
        near-nucleus, far region, and full oh column density curve
    """

    continuum_keys = [
        ContinuumSubtractionKey(
            epoch_id=eid.epoch_id,
            oh_filter=oh_filter,
            dust_filter=dust_filter,
            stacking_method=stacking_method,
            dust_redness_pct_per_hundred_nm=x,
        )
        for x in scp.dust_rednesses
    ]

    all_redness_results = [
        scp.load_radial_profile_water_production_analysis(key=x) for x in continuum_keys
    ]
    all_redness_result_dfs = [
        dataframe_from_radial_water_production_analysis(
            rpwpa=x, dust_redness=k.dust_redness_pct_per_hundred_nm
        )
        for x, k in zip(all_redness_results, continuum_keys)
    ]
    valid_results: list[pd.DataFrame] = list(filter(lambda x: x is not None, all_redness_result_dfs))  # type: ignore
    if len(valid_results) == 0:
        print(
            f"No valid found while assembling vectorial water production results for {eid.epoch_id}: {oh_filter=}, {dust_filter=}, {stacking_method=}"
        )
        return pd.DataFrame()

    valid_df = pd.concat(valid_results).reset_index(drop=True)

    return add_epoch_index_entry_to_dataframe(df=valid_df, eid=eid)
