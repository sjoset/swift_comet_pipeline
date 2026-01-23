from swift_comet_pipeline.photometry.background.determine_background import (
    determine_background,
)
from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import EpochSubpipelineKey
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.scp_types.primitive import *


def do_background_determination(scp: Products, ref: ProductReference) -> None:

    # TODO: log failures before returning

    pkey = ref.key
    assert isinstance(pkey, EpochSubpipelineKey)

    stack_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background, key=pkey
    )
    stack_fits = scp.load_fits_image(ref=stack_ref)
    if not stack_fits:
        # print(f"Could not load image for product {stack_ref}!  Skipping.")
        return
    stack_img = stack_fits.data
    assert isinstance(stack_img, SwiftUvotImage)

    exp_ref = ProductReference(kind=ProductKind.stacked_image_exposure_map, key=pkey)
    exp_fits = scp.load_fits_image(ref=exp_ref)
    if not exp_fits:
        # print(f"Could not exposure map for {exp_ref}!  Skipping.")
        return
    exp_map = exp_fits.data
    assert isinstance(exp_map, SwiftUvotImage)

    bg_result = determine_background(
        img=stack_img,
        exposure_map=exp_map,
        filter_type=pkey.filter_type,
        epoch_id=pkey.epoch_id,
    )

    if not bg_result:
        # print(f"Could not get background result for {ref}!")
        return

    scp.save_background_result(bg_result=bg_result, key=pkey)
