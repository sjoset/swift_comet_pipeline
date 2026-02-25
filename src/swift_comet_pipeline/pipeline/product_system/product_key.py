from dataclasses import dataclass

from swift_comet_pipeline.scp_types.primitive import *


# -----------------------------------------------------------------------------
# Keys (pure data; no path/formatting behavior)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyLike:
    pass


@dataclass(frozen=True)
class GlobalKey(KeyLike):
    pass

    def __str__(self):
        return ""


@dataclass(frozen=True)
class EpochSubpipelineKey(KeyLike):
    epoch_id: EpochID
    filter_type: UvotFilter
    stacking_method: StackingMethod

    def __str__(self):
        return f"epoch id: {self.epoch_id}  filter: {self.filter_type} stacking: {self.stacking_method}"


@dataclass(frozen=True)
class ContinuumSubtractionKey(KeyLike):
    epoch_id: EpochID
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod
    dust_redness_pct_per_hundred_nm: float

    def __str__(self):
        return f"{self.epoch_id} oh: {self.oh_filter} dust: {self.dust_filter} redness: {self.dust_redness_pct_per_hundred_nm} stacking: {self.stacking_method}"


# contains results for every redness and every epoch
@dataclass(frozen=True)
class LightcurveKey(KeyLike):
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod

    def __str__(self):
        return f"lightcurve for oh: {self.oh_filter} dust: {self.dust_filter} stacking: {self.stacking_method}"


# contains blue spot results for every redness and every epoch
@dataclass(frozen=True)
class BlueSpotLightcurveKey(KeyLike):
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod
    blue_spot_extent_km: float

    def __str__(self):
        return f"blue spot lightcurve for oh: {self.oh_filter} dust: {self.dust_filter} stacking: {self.stacking_method} extent: {self.blue_spot_extent_km} km"


# contains results for one dust prior sigma, over all dust rednesses
@dataclass(frozen=True)
class BayesianPriorLightcurveKey(KeyLike):
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod
    # Parameter for our Gaussian prior on the dust redness
    dust_redness_sigma_pct_per_hundred_nm: float

    def __str__(self):
        return f"bayesian lightcurve for oh: {self.oh_filter} dust: {self.dust_filter} prior sigma: {self.dust_redness_sigma_pct_per_hundred_nm} {self.stacking_method}"


# contains results for one dust prior sigma and one blue spot extent, over all dust rednesses
@dataclass(frozen=True)
class BayesianPriorBlueSpotLightcurveKey(KeyLike):
    oh_filter: UvotFilter
    dust_filter: UvotFilter
    stacking_method: StackingMethod
    blue_spot_extent_km: float
    # Parameter for our Gaussian prior on the dust redness
    dust_redness_sigma_pct_per_hundred_nm: float

    def __str__(self):
        return f"bayesian blue spot lightcurve for oh: {self.oh_filter} dust: {self.dust_filter} prior sigma: {self.dust_redness_sigma_pct_per_hundred_nm} stacking: {self.stacking_method} extent: {self.blue_spot_extent_km} km"
