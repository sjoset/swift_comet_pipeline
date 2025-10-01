from swift_comet_pipeline.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.types.comet_profile import CometRadialProfile
from swift_comet_pipeline.types.dust_reddening_percent import DustReddeningPercent


def subtract_profiles(
    uw1_profile: CometRadialProfile,
    uvv_profile: CometRadialProfile,
    dust_redness: DustReddeningPercent,
) -> CometRadialProfile:
    # TODO: documentation

    # function assumes radial profile from both filters is the same length radially
    assert len(uw1_profile.profile_axis_xs) == len(uvv_profile.profile_axis_xs)

    # should all be zero - the radial axes should be sampled the same way
    # print(uw1_profile.profile_axis_xs - uvv_profile.profile_axis_xs)

    beta = beta_parameter(dust_redness)

    subtracted_pixels = uw1_profile.pixel_values - beta * uvv_profile.pixel_values

    assert uw1_profile._cone_size == uvv_profile._cone_size

    return CometRadialProfile(
        profile_axis_xs=uw1_profile.profile_axis_xs,
        pixel_values=subtracted_pixels,
        _xs=uw1_profile._xs,
        _ys=uw1_profile._ys,
        _radius=uw1_profile._radius,
        _theta=uw1_profile._theta,
        _cone_size=uw1_profile._cone_size,
        _comet_center=uw1_profile._comet_center,
    )
