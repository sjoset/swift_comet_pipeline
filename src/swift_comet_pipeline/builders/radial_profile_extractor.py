from swift_comet_pipeline.pipeline.product_system.product_facade import Products
from swift_comet_pipeline.pipeline.product_system.product_key import EpochSubpipelineKey
from swift_comet_pipeline.pipeline.product_system.product_reference import (
    ProductReference,
)
from swift_comet_pipeline.ui.mpl_ui.mpl_ui_radial_profile_from_cone import (
    profile_extraction_from_cone,
)


def do_radial_profile_from_cone(scp: Products, ref: ProductReference) -> None:
    crpfc = profile_extraction_from_cone(scp=scp, ref=ref)

    assert isinstance(ref.key, EpochSubpipelineKey)
    scp.save_extracted_radial_profile(crp=crpfc, key=ref.key)
