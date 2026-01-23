from functools import partial
from pathlib import Path

from astropy.io import fits

from swift_comet_pipeline.data_ingestion.epoch_index.find_epoch_index_entry import (
    get_epoch_index_entry,
)
from swift_comet_pipeline.pipeline.product_system.product_key import (
    BayesianPriorBlueSpotLightcurveKey,
    BayesianPriorLightcurveKey,
    BlueSpotLightcurveKey,
    ContinuumSubtractionKey,
    EpochSubpipelineKey,
    LightcurveKey,
)
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.pipeline.product_system.registry_builder import (
    add_continuum_subtraction_products_to_registry,
    add_epoch_subpipelines_to_registry,
    add_lightcurve_products_to_registry,
    data_ingestion_registry,
)
from swift_comet_pipeline.scp_types.compound.background_result import (
    BackgroundResult,
    background_result_from_json,
    json_from_background_result,
)
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfileFromConicalRegion,
    comet_radial_profile_from_cone_from_json,
    json_from_comet_radial_profile_from_cone,
)
from swift_comet_pipeline.scp_types.compound.epoch_index import (
    EpochIndex,
    EpochIndexEntry,
    epoch_index_from_json,
    json_from_epoch_index,
)
from swift_comet_pipeline.scp_types.compound.lightcurve import (
    BayesianPriorBlueSpotLightCurve,
    BlueSpotLightCurve,
    LightCurve,
    bayesian_prior_blue_spot_lightcurve_to_dataframe,
    blue_spot_lightcurve_to_dataframe,
    dataframe_to_bayesian_prior_blue_spot_lightcurve,
    dataframe_to_blue_spot_lightcurve,
    dataframe_to_lightcurve,
    lightcurve_to_dataframe,
)
from swift_comet_pipeline.scp_types.compound.radial_profile_water_production import (
    RadialProfileWaterProductionAnalysis,
    json_from_radial_profile_water_production_analysis,
    radial_profile_water_production_analysis_from_json,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.scp_types.compound.comet_project_config import (
    CometProjectConfig,
)
from swift_comet_pipeline.scp_types.primitive.afrho_from_profile import (
    AfrhoFromRadialProfile,
    afrho_from_radial_profile_from_dataframe,
    dataframe_from_afrho_from_radial_profile,
)


# -----------------------------------------------------------------------------
# Convenience facade that binds a project config and loads products from it
# -----------------------------------------------------------------------------
class Products:
    """
    Facade to simplify dealing with product storage and retrieval
    This should be regenerated when products are deleted/added.
    """

    def __init__(self, cfg: CometProjectConfig):
        self.cfg = cfg
        dust_rednesses = [
            DustReddeningPercent(x)
            for x in np.arange(
                cfg.dust_redness_min,
                cfg.dust_redness_max + cfg.dust_redness_step,
                step=cfg.dust_redness_step,
            )
        ]
        self.dust_rednesses = dust_rednesses
        blue_spot_extents_km = np.arange(
            cfg.blue_spot_extent_km_min,
            cfg.blue_spot_extent_km_max + cfg.blue_spot_extent_km_step,
            step=cfg.blue_spot_extent_km_step,
        )
        self.blue_spot_extents_km = [float(x) for x in blue_spot_extents_km]
        self._generate_registry()

    def _generate_registry(self):
        self.reg = data_ingestion_registry()
        self.registry_load = partial(self.reg.load, cfg=self.cfg)
        self.registry_save = partial(self.reg.save, cfg=self.cfg)

        self.epoch_index = self.load_epoch_index()
        if self.epoch_index is None:
            return
        add_epoch_subpipelines_to_registry(reg=self.reg, epoch_index=self.epoch_index)
        add_continuum_subtraction_products_to_registry(
            reg=self.reg,
            epoch_index=self.epoch_index,
            oh_filters=self.cfg.oh_filters,
            dust_filters=self.cfg.dust_filters,
            dust_rednesses=self.dust_rednesses,
        )
        add_lightcurve_products_to_registry(
            reg=self.reg,
            epoch_index=self.epoch_index,
            oh_filters=self.cfg.oh_filters,
            dust_filters=self.cfg.dust_filters,
            dust_rednesses=self.dust_rednesses,
            bayesian_prior_sigmas=self.cfg.bayesian_prior_sigmas,
            blue_spot_extents_km=self.blue_spot_extents_km,
        )

    def regenerate(self):
        self._generate_registry()

    def exists(self, ref: ProductReference) -> bool:
        return self.reg.exists(ref=ref, cfg=self.cfg)

    def path_for(self, ref: ProductReference) -> Path | None:
        return self.reg.path_for(ref=ref, cfg=self.cfg)

    def load_raw_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.registry_load(
            ProductReference(kind=ProductKind.observation_log_raw)
        )

    def save_raw_log(self, df: SwiftUvotObservationLogDataframe) -> Path | None:
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.observation_log_raw), obj=df
        )

    def load_epoch_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.registry_load(
            ref=ProductReference(kind=ProductKind.observation_log_with_epochs)
        )

    def save_epoch_log(self, df: pd.DataFrame) -> Path | None:
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.observation_log_with_epochs), obj=df
        )

    def load_obs_log(self) -> SwiftUvotObservationLogDataframe | None:
        return self.registry_load(
            ref=ProductReference(kind=ProductKind.observation_log_with_vetoes)
        )

    def save_obs_log(self, df: pd.DataFrame) -> Path | None:
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.observation_log_with_vetoes), obj=df
        )

    def load_earth_orbit_data(self) -> pd.DataFrame | None:
        return self.registry_load(
            ref=ProductReference(kind=ProductKind.orbit_data_earth)
        )

    def save_earth_orbit_data(self, df: pd.DataFrame) -> Path | None:
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.orbit_data_earth), obj=df
        )

    def load_comet_orbit_data(self) -> pd.DataFrame | None:
        return self.registry_load(
            ref=ProductReference(kind=ProductKind.orbit_data_comet)
        )

    def save_comet_orbit_data(self, df: pd.DataFrame) -> Path | None:
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.orbit_data_comet), obj=df
        )

    # Epoch index
    def load_epoch_index(self):
        json_dict = self.registry_load(
            ref=ProductReference(kind=ProductKind.epoch_index)
        )
        if json_dict is None:
            return None
        return epoch_index_from_json(json_dict=json_dict)

    def load_epoch_index_entry(self, epoch_id: EpochID) -> EpochIndexEntry | None:
        epoch_index = self.load_epoch_index()
        assert epoch_index is not None
        return get_epoch_index_entry(epoch_index=epoch_index, epoch_id=epoch_id)

    def save_epoch_index(self, epoch_index: EpochIndex) -> Path | None:
        json_dict = json_from_epoch_index(epoch_index=epoch_index)
        return self.registry_save(
            ref=ProductReference(kind=ProductKind.epoch_index), obj=json_dict
        )

    # FITS images
    def load_fits_image(self, ref: ProductReference) -> fits.ImageHDU | None:
        return self.registry_load(ref=ref)

    def save_fits_image(self, img: fits.ImageHDU, ref: ProductReference):
        return self.registry_save(ref=ref, obj=img)
        # return self.reg.save(ref=ref, obj=img, cfg=self.cfg)

    # background
    def load_background_result(
        self, key: EpochSubpipelineKey
    ) -> BackgroundResult | None:
        pref = ProductReference(kind=ProductKind.background_determination, key=key)
        json_dict = self.registry_load(ref=pref)
        return background_result_from_json(json_dict=json_dict)

    def save_background_result(
        self, bg_result: BackgroundResult, key: EpochSubpipelineKey
    ):
        pref = ProductReference(kind=ProductKind.background_determination, key=key)
        json_dict = json_from_background_result(bgr=bg_result)
        return self.registry_save(ref=pref, obj=json_dict)

    # aperture analysis
    def load_annular_aperture_analysis(
        self, key: EpochSubpipelineKey
    ) -> tuple[AnnularAperturePhotometryAnalysis, dict] | None:
        # returns metadata associated with the photometry as a dict
        pref = ProductReference(
            kind=ProductKind.annular_aperture_photometry_analysis, key=key
        )
        df = self.registry_load(ref=pref)
        aapa = annular_aperture_photometry_analysis_from_dataframe(df=df)
        return aapa, df.attrs

    def save_annular_aperture_analysis(
        self,
        aapa: AnnularAperturePhotometryAnalysis,
        metadata: dict,
        key: EpochSubpipelineKey,
    ) -> pathlib.Path | None:
        pref = ProductReference(
            kind=ProductKind.annular_aperture_photometry_analysis, key=key
        )
        aapa_df = dataframe_from_annular_aperture_photometry_analysis(aapa=aapa)
        aapa_df.attrs = metadata
        return self.registry_save(ref=pref, obj=aapa_df)

    # extracted radial profiles
    def load_extracted_radial_profile(
        self, key: EpochSubpipelineKey
    ) -> CometRadialProfileFromConicalRegion:
        pref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=key)
        json_dict = self.registry_load(ref=pref)
        return comet_radial_profile_from_cone_from_json(json_dict=json_dict)

    def save_extracted_radial_profile(
        self, crp: CometRadialProfileFromConicalRegion, key: EpochSubpipelineKey
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.radial_profile_from_cone, key=key)
        json_dict = json_from_comet_radial_profile_from_cone(crpfc=crp)
        return self.registry_save(obj=json_dict, ref=pref)

    # aperture water analysis
    def load_aperture_water_production_analysis(
        self, key: ContinuumSubtractionKey
    ) -> ApertureWaterProductionAnalysis:
        pref = ProductReference(kind=ProductKind.aperture_water_production, key=key)
        df = self.registry_load(ref=pref)
        return aperture_water_production_analysis_from_dataframe(df=df)

    def save_aperture_water_production_analysis(
        self, awpa: ApertureWaterProductionAnalysis, key: ContinuumSubtractionKey
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.aperture_water_production, key=key)
        awpa_df = dataframe_from_aperture_water_production_analysis(awpa=awpa)
        return self.registry_save(ref=pref, obj=awpa_df)

    # radial profile water production
    def load_radial_profile_water_production_analysis(
        self, key: ContinuumSubtractionKey
    ) -> RadialProfileWaterProductionAnalysis:
        pref = ProductReference(
            kind=ProductKind.radial_profile_water_production, key=key
        )
        rpwpa_json_dict = self.registry_load(ref=pref)
        return radial_profile_water_production_analysis_from_json(
            json_dict=rpwpa_json_dict
        )

    def save_radial_profile_water_production_analysis(
        self, rpwpa: RadialProfileWaterProductionAnalysis, key: ContinuumSubtractionKey
    ) -> pathlib.Path | None:
        pref = ProductReference(
            kind=ProductKind.radial_profile_water_production, key=key
        )
        rpwpa_json_dict = json_from_radial_profile_water_production_analysis(
            rpwpa=rpwpa
        )
        return self.registry_save(ref=pref, obj=rpwpa_json_dict)

    # afrho from apertures
    def load_afrho_from_aperture_photometry(
        self, key: EpochSubpipelineKey
    ) -> AfrhoFromAperturePhotometryAnalysis:
        pref = ProductReference(
            kind=ProductKind.afrho_from_aperture_photometry_analysis, key=key
        )
        afapa_df = self.registry_load(ref=pref)
        return afrho_aperture_photometry_analysis_from_dataframe(df=afapa_df)

    def save_afrho_from_aperture_photometry(
        self, afapa: AfrhoFromAperturePhotometryAnalysis, key: EpochSubpipelineKey
    ) -> pathlib.Path | None:
        pref = ProductReference(
            kind=ProductKind.afrho_from_aperture_photometry_analysis, key=key
        )
        afapa_df = dataframe_from_afrho_aperture_photometry_analysis(afapa=afapa)
        return self.registry_save(ref=pref, obj=afapa_df)

    # afrho from profiles
    def load_afrho_from_profile(
        self, key: EpochSubpipelineKey
    ) -> AfrhoFromRadialProfile:
        pref = ProductReference(kind=ProductKind.afrho_from_radial_profile, key=key)
        afrp_df = self.registry_load(ref=pref)
        return afrho_from_radial_profile_from_dataframe(df=afrp_df)

    def save_afrho_from_profile(
        self, afrp: AfrhoFromRadialProfile, key: EpochSubpipelineKey
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.afrho_from_radial_profile, key=key)
        afrp_df = dataframe_from_afrho_from_radial_profile(afrp=afrp)
        return self.registry_save(ref=pref, obj=afrp_df)

    def load_water_production_lightcurve(self, key: LightcurveKey) -> LightCurve:
        pref = ProductReference(kind=ProductKind.water_production_lightcurve, key=key)
        water_df = self.registry_load(ref=pref)
        water_lc = dataframe_to_lightcurve(df=water_df)
        return water_lc

    def save_water_production_lightcurve(
        self, water_lc: LightCurve, key: LightcurveKey
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.water_production_lightcurve, key=key)
        water_df = lightcurve_to_dataframe(lc=water_lc)
        return self.registry_save(ref=pref, obj=water_df)

    def load_bayesian_water_production_lightcurve(
        self, key: BayesianPriorLightcurveKey
    ) -> LightCurve:
        pref = ProductReference(
            kind=ProductKind.bayesian_water_production_lightcurve, key=key
        )
        water_df = self.registry_load(ref=pref)
        water_lc = dataframe_to_lightcurve(df=water_df)
        return water_lc

    def save_bayesian_water_production_lightcurve(
        self, water_lc: LightCurve, key: BayesianPriorLightcurveKey
    ) -> pathlib.Path | None:
        pref = ProductReference(
            kind=ProductKind.bayesian_water_production_lightcurve, key=key
        )
        water_df = lightcurve_to_dataframe(lc=water_lc)
        return self.registry_save(ref=pref, obj=water_df)

    def load_blue_spot_lightcurve(
        self, key: BlueSpotLightcurveKey
    ) -> BlueSpotLightCurve:
        pref = ProductReference(kind=ProductKind.blue_spot_lightcurve, key=key)
        bs_df = self.registry_load(ref=pref)
        bs_lc = dataframe_to_blue_spot_lightcurve(df=bs_df)
        return bs_lc

    def save_blue_spot_lightcurve(
        self, bs_lc: BlueSpotLightCurve, key: BlueSpotLightcurveKey
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.blue_spot_lightcurve, key=key)
        bs_df = blue_spot_lightcurve_to_dataframe(lc=bs_lc)
        return self.registry_save(ref=pref, obj=bs_df)

    def load_bayesian_blue_spot_lightcurve(
        self, key: BayesianPriorBlueSpotLightcurveKey
    ) -> BayesianPriorBlueSpotLightCurve:
        pref = ProductReference(kind=ProductKind.bayesian_blue_spot_lightcurve, key=key)
        bs_df = self.registry_load(ref=pref)
        bs_lc = dataframe_to_bayesian_prior_blue_spot_lightcurve(df=bs_df)
        return bs_lc

    def save_bayesian_prior_blue_spot_lightcurve(
        self,
        bs_lc: BayesianPriorBlueSpotLightCurve,
        key: BayesianPriorBlueSpotLightcurveKey,
    ) -> pathlib.Path | None:
        pref = ProductReference(kind=ProductKind.bayesian_blue_spot_lightcurve, key=key)
        bs_df = bayesian_prior_blue_spot_lightcurve_to_dataframe(lc=bs_lc)
        return self.registry_save(ref=pref, obj=bs_df)
