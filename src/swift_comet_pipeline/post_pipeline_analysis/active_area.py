import numpy as np
import pandas as pd
import astropy.units as u

from swift_comet_pipeline.dust.reddening_correction import DustReddeningPercent
from swift_comet_pipeline.lightcurve.lightcurve_bayesian import (
    BayesianLightCurve,
    bayesian_lightcurve_to_dataframe,
)
from swift_comet_pipeline.water_production.active_area import estimate_active_area


def calculate_active_area_from_bayesian_lightcurve(
    blc: BayesianLightCurve,
    dust_mean: DustReddeningPercent,
    dust_sigma: float,
    lower_assumed_comet_radius: u.Quantity,
    upper_assumed_comet_radius: u.Quantity,
) -> pd.DataFrame:
    """
    Takes a BayesianLightCurve and returns the lightcurve as a dataframe with extra columns:

    active_area_perp: active area calculated with the rotation axis pointed perpendicular to sun-comet axis, astropy units attached
    active_area_par: active area calculated with the rotation axis pointed parallel to sun-comet axis, astropy units attached

    active_area_perp_km2: as above, but floating point in km2
    active_area_par_km2: as above, but floating point in km2

    active_fraction_perp_lower: active fraction of the nuclear area in the perpendicular orientation, using the lower assumed comet radius
    active_fraction_perp_upper: active fraction of the nuclear area in the perpendicular orientation, using the upper assumed comet radius

    active_fraction_par_lower: as above, with parallel orientation and lower assumed comet radius
    active_fraction_par_upper: as above, with parallel orientation and upper assumed comet radius
    """

    lower_assumed_comet_radius_km: float = lower_assumed_comet_radius.to_value(u.km)  # type: ignore
    upper_assumed_comet_radius_km: float = upper_assumed_comet_radius.to_value(u.km)  # type: ignore

    blc_df_base = bayesian_lightcurve_to_dataframe(blc=blc)

    blc_df = blc_df_base[
        (blc_df_base.dust_mean == dust_mean) & (blc_df_base.dust_sigma == dust_sigma)
    ].copy()

    blc_df["active_area_perp"] = blc_df.apply(
        lambda row: estimate_active_area(
            q=row.posterior_q / u.s,  # type: ignore
            rh=row.rh_au * u.AU,  # type: ignore
            sub_solar_latitude=0 * u.degree,  # type: ignore
        ),
        axis=1,
    )
    blc_df["active_area_par"] = blc_df.apply(
        lambda row: estimate_active_area(
            q=row.posterior_q / u.s,  # type: ignore
            rh=row.rh_au * u.AU,  # type: ignore
            sub_solar_latitude=90 * u.degree,  # type: ignore
        ),
        axis=1,
    )

    blc_df["active_area_perp_km2"] = blc_df.apply(
        lambda row: row.active_area_perp.to_value(u.km**2), axis=1  # type: ignore
    )
    blc_df["active_fraction_perp_lower"] = blc_df.active_area_perp_km2 / (
        4 * np.pi * lower_assumed_comet_radius_km**2
    )
    blc_df["active_fraction_perp_upper"] = blc_df.active_area_perp_km2 / (
        4 * np.pi * upper_assumed_comet_radius_km**2
    )

    blc_df["active_area_par_km2"] = blc_df.apply(
        lambda row: row.active_area_par.to_value(u.km**2), axis=1  # type: ignore
    )
    blc_df["active_fraction_par_lower"] = blc_df.active_area_par_km2 / (
        4 * np.pi * lower_assumed_comet_radius_km**2
    )
    blc_df["active_fraction_par_upper"] = blc_df.active_area_par_km2 / (
        4 * np.pi * upper_assumed_comet_radius_km**2
    )

    return blc_df  # type: ignore
