from swift_comet_pipeline.photometry.dust.beta_parameter import beta_parameter
from swift_comet_pipeline.scp_types.compound.comet_profile import (
    CometRadialProfile,
    CometRadialProfileFromConicalRegion,
)
from swift_comet_pipeline.scp_types.primitive import *


# TODO: change this to take CometRadialProfile if we don't care about angle
# or split into two functions, one for each type of radial profile


def subtract_profiles(
    oh_profile: CometRadialProfileFromConicalRegion,
    dust_profile: CometRadialProfileFromConicalRegion,
    dust_redness: DustReddeningPercent,
    oh_filter: UvotFilter,
    dust_filter: UvotFilter,
) -> CometRadialProfile:
    # TODO: documentation

    # This function assumes that the pixel scale does not change considerably during an epoch

    # if the profiles are unequal lengths, use the shorter profile length for subtraction
    subtraction_profile_len = min(
        len(oh_profile.profile_axis_rs), len(dust_profile.profile_axis_rs)
    )

    # # function assumes radial profile from both filters is the same length radially
    # assert len(oh_profile.profile_axis_rs) == len(dust_profile.profile_axis_rs)

    # if oh_profile._theta != dust_profile._theta:
    #     print("Warning: subtracting profiles taken at different angles!")

    beta = beta_parameter(
        dust_redness=dust_redness, oh_filter=oh_filter, dust_filter=dust_filter
    )
    subtracted_pixels = (
        oh_profile.pixel_values[:subtraction_profile_len]
        - beta * dust_profile.pixel_values[:subtraction_profile_len]
    )
    subtracted_profile_rs = oh_profile.profile_axis_rs[:subtraction_profile_len]

    # # TODO: not necessary
    # assert oh_profile._cone_size == dust_profile._cone_size

    return CometRadialProfile(
        profile_axis_rs=subtracted_profile_rs,
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
