from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import EpochSubpipelineKey
from swift_comet_pipeline.pipeline.product_system.product_kind import ProductKind
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.scp_types.primitive.stacking_method import StackingMethod
from swift_comet_pipeline.stacking.stacking import do_stacking


def do_stack(scp: Products, ref: ProductReference) -> None:

    assert isinstance(ref.key, EpochSubpipelineKey)
    sum_med_exp = do_stacking(scp=scp, key=ref.key)

    if sum_med_exp is None:
        # TODO: log here or otherwise signal falure
        # print("Stacking failed!")
        return

    img_sum, img_median, img_exposure_map = sum_med_exp

    # TODO: show the images and ask whether we want to save or not
    sum_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.summation,
        ),
    )
    med_ref = ProductReference(
        kind=ProductKind.stacked_image_with_background,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.median,
        ),
    )
    exposure_sum_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.summation,
        ),
    )
    exposure_med_ref = ProductReference(
        kind=ProductKind.stacked_image_exposure_map,
        key=EpochSubpipelineKey(
            epoch_id=ref.key.epoch_id,
            filter_type=ref.key.filter_type,
            stacking_method=StackingMethod.median,
        ),
    )
    scp.save_fits_image(img=img_sum, ref=sum_ref)
    scp.save_fits_image(img=img_median, ref=med_ref)
    scp.save_fits_image(img=img_exposure_map, ref=exposure_sum_ref)
    scp.save_fits_image(img=img_exposure_map, ref=exposure_med_ref)
