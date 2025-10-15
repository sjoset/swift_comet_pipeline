from swift_comet_pipeline.scp_types.primitive import *
from swift_comet_pipeline.pipeline.product_system.registry_and_store import (
    EpochSubpipelineKey,
    ProductKind,
    ProductReference,
    Products,
)


def do_background_subtraction(scp: Products, ref: ProductReference) -> None:

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    stack_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background, key=pkey
    )
    stack_fits = scp.load_fits_image(ref=stack_ref)
    if stack_fits is None:
        print(f"Could not load image for product {stack_ref}!  Skipping.")
        return
    stack_img = stack_fits.data
    assert isinstance(stack_img, SwiftUvotImage)

    bgr = scp.load_background_result(key=pkey)
    if bgr is None:
        print(
            f"Could not load the background determination for {stack_ref}!  Skipping."
        )
        return

    bg_sub_fits = stack_fits.copy()
    bg_sub_fits.data = stack_img - bgr.b_hat
    bg_sub_fits.header["bg_subtracted"] = True

    bg_sub_ref = ProductReference(
        kind=ProductKind.bg_subtracted_stacked_image, key=pkey
    )
    scp.save_fits_image(img=bg_sub_fits, ref=bg_sub_ref)
