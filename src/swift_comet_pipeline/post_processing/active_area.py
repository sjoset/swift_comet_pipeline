import pathlib

import numpy as np
import pandas as pd
import astropy.units as u

from swift_comet_pipeline.lightcurve.lightcurve_bayesian import (
    BayesianLightCurve,
    bayesian_lightcurve_to_dataframe,
)
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.pipeline import SwiftCometPipeline
from swift_comet_pipeline.post_processing.post_processing_steps import (
    EpochPostProcessingStep,
    apply_epoch_post_processing_pipeline,
)
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.water_production.active_area import estimate_active_area


__all__ = ["calculate_active_area_from_bayesian_lightcurve"]


class EstimateActiveArea(EpochPostProcessingStep):
    """
    Fills the given column name with the active area as a float, after converting to area measured in (area_units)**2,
    using the sublimation model with the tilt of the comet's axis described through sub_solar_latitude
    """

    def __init__(
        self,
        active_area_column_name: str,
        area_units: u.Quantity,
        sub_solar_latitude: u.Quantity,
    ):
        super().__init__()
        self.sub_solar_latitude = sub_solar_latitude
        self.active_area_column_name = active_area_column_name
        self.area_units = area_units

        self.required_input_columns = ["posterior_q", "rh_au"]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.active_area_column_name] = df.apply(
            lambda row: estimate_active_area(
                q=row.posterior_q / u.s,  # type: ignore
                rh=row.rh_au * u.AU,  # type: ignore
                sub_solar_latitude=self.sub_solar_latitude,
            ),
            axis=1,
        )
        df[self.active_area_column_name] = df.apply(
            lambda row: row[self.active_area_column_name].to_value(self.area_units**2),
            axis=1,
        )
        return df


class CalculateActiveAreaFraction(EpochPostProcessingStep):
    def __init__(
        self,
        active_area_source_column: str,
        active_fraction_column: str,
        area_units: u.Quantity,
        comet_radius: u.Quantity,
    ):
        super().__init__()
        self.active_area_source_column = active_area_source_column
        self.active_fraction_column = active_fraction_column
        self.area_units = area_units
        self.comet_radius = comet_radius
        self.comet_area = 4 * np.pi * self.comet_radius.to_value(self.area_units)  # type: ignore

        self.required_input_columns = [self.active_area_source_column]

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        if not self.check_required_columns(
            df_in=df_in, step_name=self.__class__.__name__
        ):
            return df_in

        df = df_in.copy()
        df[self.active_fraction_column] = (
            df[self.active_area_source_column] / self.comet_radius
        )
        return df


def calculate_active_area_from_bayesian_lightcurve(
    scp: SwiftCometPipeline,
    stacking_method: StackingMethod,
    comet_radius_lower: u.Quantity,
    comet_radius_upper: u.Quantity,
) -> pd.DataFrame | None:

    bayes_df = scp.get_product_data(
        pf=PipelineFilesEnum.bayesian_aperture_lightcurve,
        stacking_method=stacking_method,
    )
    if bayes_df is None:
        return None

    active_area_pipeline_steps = [
        EstimateActiveArea(
            active_area_column_name="active_area_perp_km2",
            area_units=u.km,  # type: ignore
            sub_solar_latitude=0 * u.degree,  # type: ignore
        ),
        CalculateActiveAreaFraction(
            active_area_source_column="active_area_perp_km2",
            active_fraction_column="active_fraction_perp_lower_radius",
            area_units=u.km,  # type: ignore
            comet_radius=comet_radius_lower,
        ),
        CalculateActiveAreaFraction(
            active_area_source_column="active_area_perp_km2",
            active_fraction_column="active_fraction_perp_upper_radius",
            area_units=u.km,  # type: ignore
            comet_radius=comet_radius_upper,
        ),
        EstimateActiveArea(
            active_area_column_name="active_area_par_km2",
            area_units=u.km,  # type: ignore
            sub_solar_latitude=0 * u.degree,  # type: ignore
        ),
        CalculateActiveAreaFraction(
            active_area_source_column="active_area_par_km2",
            active_fraction_column="active_fraction_par_lower_radius",
            area_units=u.km,  # type: ignore
            comet_radius=comet_radius_lower,
        ),
        CalculateActiveAreaFraction(
            active_area_source_column="active_area_par_km2",
            active_fraction_column="active_fraction_par_upper_radius",
            area_units=u.km,  # type: ignore
            comet_radius=comet_radius_upper,
        ),
    ]

    # TODO: enable post-process cache
    results_cache_path = (
        pathlib.Path(scp.pipeline_files.base_project_path)
        / "post_processing"
        / "active_area_bayes.csv"
    )

    df = apply_epoch_post_processing_pipeline(
        initial_dataframe=bayes_df,
        ep=active_area_pipeline_steps,
        # results_cache_path=results_cache_path,
    )

    return df


# TODO: remove old code

# def calculate_active_area_from_bayesian_lightcurve_old(
#     blc: BayesianLightCurve,
#     dust_mean: DustReddeningPercent,
#     dust_sigma: float,
#     lower_assumed_comet_radius: u.Quantity,
#     upper_assumed_comet_radius: u.Quantity,
# ) -> pd.DataFrame:
#     """
#     Takes a BayesianLightCurve and returns the lightcurve as a dataframe with extra columns:
#
#     active_area_perp: active area calculated with the rotation axis pointed perpendicular to sun-comet axis, astropy units attached
#     active_area_par: active area calculated with the rotation axis pointed parallel to sun-comet axis, astropy units attached
#
#     active_area_perp_km2: as above, but floating point in km2
#     active_area_par_km2: as above, but floating point in km2
#
#     active_fraction_perp_lower: active fraction of the nuclear area in the perpendicular orientation, using the lower assumed comet radius
#     active_fraction_perp_upper: active fraction of the nuclear area in the perpendicular orientation, using the upper assumed comet radius
#
#     active_fraction_par_lower: as above, with parallel orientation and lower assumed comet radius
#     active_fraction_par_upper: as above, with parallel orientation and upper assumed comet radius
#     """
#
#     lower_assumed_comet_radius_km: float = lower_assumed_comet_radius.to_value(u.km)  # type: ignore
#     upper_assumed_comet_radius_km: float = upper_assumed_comet_radius.to_value(u.km)  # type: ignore
#
#     blc_df_base = bayesian_lightcurve_to_dataframe(blc=blc)
#
#     blc_df = blc_df_base[
#         (blc_df_base.dust_mean == dust_mean) & (blc_df_base.dust_sigma == dust_sigma)
#     ].copy()
#
#     blc_df["active_area_perp"] = blc_df.apply(
#         lambda row: estimate_active_area(
#             q=row.posterior_q / u.s,  # type: ignore
#             rh=row.rh_au * u.AU,  # type: ignore
#             sub_solar_latitude=0 * u.degree,  # type: ignore
#         ),
#         axis=1,
#     )
#     blc_df["active_area_par"] = blc_df.apply(
#         lambda row: estimate_active_area(
#             q=row.posterior_q / u.s,  # type: ignore
#             rh=row.rh_au * u.AU,  # type: ignore
#             sub_solar_latitude=90 * u.degree,  # type: ignore
#         ),
#         axis=1,
#     )
#
#     blc_df["active_area_perp_km2"] = blc_df.apply(
#         lambda row: row.active_area_perp.to_value(u.km**2), axis=1  # type: ignore
#     )
#     blc_df["active_fraction_perp_lower"] = blc_df.active_area_perp_km2 / (
#         4 * np.pi * lower_assumed_comet_radius_km**2
#     )
#     blc_df["active_fraction_perp_upper"] = blc_df.active_area_perp_km2 / (
#         4 * np.pi * upper_assumed_comet_radius_km**2
#     )
#
#     blc_df["active_area_par_km2"] = blc_df.apply(
#         lambda row: row.active_area_par.to_value(u.km**2), axis=1  # type: ignore
#     )
#     blc_df["active_fraction_par_lower"] = blc_df.active_area_par_km2 / (
#         4 * np.pi * lower_assumed_comet_radius_km**2
#     )
#     blc_df["active_fraction_par_upper"] = blc_df.active_area_par_km2 / (
#         4 * np.pi * upper_assumed_comet_radius_km**2
#     )
#
#     return blc_df  # type: ignore
