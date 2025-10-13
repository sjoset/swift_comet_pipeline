from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfile,
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.primitive import *


# TODO: change this to take CometRadialProfile if we don't care about angle
# or split into two functions, one for each type of radial profile

# TODO: This fails if profiles are not the same length - is there a better way to handle it?


def subtract_profiles(
    uw1_profile: CometRadialProfileFromConicalRegion,
    uvv_profile: CometRadialProfileFromConicalRegion,
    dust_redness: DustReddeningPercent,
) -> CometRadialProfile:
    # TODO: documentation

    # function assumes radial profile from both filters is the same length radially
    assert len(uw1_profile.profile_axis_rs) == len(uvv_profile.profile_axis_rs)

    # TODO: we should resize the smaller profile with np.resize to add zeros to the end of the smaller profile

    if uw1_profile._theta != uvv_profile._theta:
        print("Warning: subtracting profiles taken at different angles!")

    beta = beta_parameter(dust_redness)

    subtracted_pixels = uw1_profile.pixel_values - beta * uvv_profile.pixel_values

    assert uw1_profile._cone_size == uvv_profile._cone_size

    return CometRadialProfile(
        profile_axis_rs=uw1_profile.profile_axis_rs,
        pixel_values=subtracted_pixels,
    )


# def subtract_profiles(
#     uw1_profile: CometRadialProfileFromConicalRegion,
#     uvv_profile: CometRadialProfileFromConicalRegion,
#     dust_redness: DustReddeningPercent,
# ) -> CometRadialProfileFromConicalRegion:
#     # TODO: documentation
#
#     # function assumes radial profile from both filters is the same length radially
#     assert len(uw1_profile.profile_axis_rs) == len(uvv_profile.profile_axis_rs)
#
#     # should all be zero - the radial axes should be sampled the same way
#     # print(uw1_profile.profile_axis_xs - uvv_profile.profile_axis_xs)
#
#     beta = beta_parameter(dust_redness)
#
#     subtracted_pixels = uw1_profile.pixel_values - beta * uvv_profile.pixel_values
#
#     assert uw1_profile._cone_size == uvv_profile._cone_size
#
#     return CometRadialProfileFromConicalRegion(
#         profile_axis_rs=uw1_profile.profile_axis_rs,
#         pixel_values=subtracted_pixels,
#         _xs=uw1_profile._xs,
#         _ys=uw1_profile._ys,
#         _radius=uw1_profile._radius,
#         _theta=uw1_profile._theta,
#         _position_angle=uw1_profile._theta
#         _cone_size=uw1_profile._cone_size,
#         _comet_center=uw1_profile._comet_center,
#     )
