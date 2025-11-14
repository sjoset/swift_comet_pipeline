from swift_comet_pipeline.photometry.comet.extract_comet_radial_profile import (
    calculate_distance_from_center_mesh,
    radial_profile_to_image,
)
from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
)


def do_radial_profile_subtraction(scp: Products, ref: ProductReference):

    assert isinstance(ref.key, EpochSubpipelineKey)

    p_key = EpochSubpipelineKey(
        epoch_id=ref.key.epoch_id,
        filter_type=ref.key.filter_type,
        stacking_method=ref.key.stacking_method,
    )

    bg_sub_ref = ProductReference(
        kind=ProductKind.bg_subtracted_stacked_image, key=p_key
    )

    # load the radial profile
    rad_prof = scp.load_extracted_radial_profile(key=p_key)

    # load the image
    img_fits = scp.load_fits_image(ref=bg_sub_ref)
    print(f"Subtracting {ref=}")
    assert img_fits is not None
    img: SwiftUvotImage = img_fits.data  # type: ignore

    mesh = calculate_distance_from_center_mesh(img=img)

    rad_prof_img = radial_profile_to_image(
        profile=rad_prof, distance_from_center_mesh=mesh
    )

    rad_sub_fits = img_fits.copy()
    rad_sub_fits.data = img - rad_prof_img
    rad_sub_fits.header["radial_profile_subtracted"] = True

    rad_sub_ref = ProductReference(
        kind=ProductKind.radial_profile_subtracted, key=ref.key
    )
    scp.save_fits_image(img=rad_sub_fits, ref=rad_sub_ref)
