from enum import StrEnum


# -----------------------------------------------------------------------------
# Product kinds (kind chooses filename stem & codec, key chooses instance)
# -----------------------------------------------------------------------------
class ProductKind(StrEnum):
    # Ingestion / logs
    observation_log_raw = "raw observation log"
    observation_log_with_epochs = "observation log with epochs"
    observation_log_with_vetoes = "observation log with vetoes"
    orbit_data_earth = "earth orbit data"
    orbit_data_comet = "comet orbit data"
    epoch_index = "epoch index"

    # -----------------
    # epoch subpipeline - for each epoch, filter, and stacking method, we have these products

    stacked_image_with_background = "stacked image, with bg"
    stacked_image_exposure_map = "stacked image exposure map"
    background_determination = "background level and error"
    bg_subtracted_stacked_image = "stacked image, no bg"

    # photometry/profiling of OH/dust
    annular_aperture_photometry_analysis = "annular aperture photometry"
    radial_profile_from_cone = "radial profile from cone"

    # radial profile subtracted image
    radial_profile_subtracted_image = "image with radial profile subtracted"

    # afrho
    afrho_from_aperture_photometry_analysis = "Afrho from aperture photometry"
    afrho_from_radial_profile = "Afrho from radial profiles"

    # -----------------
    # Water production and other final products

    # aperture continuum subtraction
    aperture_water_production = "aperture water production rate"
    # radial profile continuum subtraction
    radial_profile_water_production = "water production from vectorial fitting"

    # lightcurves
    water_production_lightcurve = "vectorial and aperture water production lightcurve"
    bayesian_water_production_lightcurve = (
        "vectorial and aperture water production lightcurve with bayesian prior"
    )
    blue_spot_lightcurve = "blue spot hydroxyl production lightcurve"
    bayesian_blue_spot_lightcurve = (
        "blue spot hydroxyl production lightcurve with bayesian prior"
    )
