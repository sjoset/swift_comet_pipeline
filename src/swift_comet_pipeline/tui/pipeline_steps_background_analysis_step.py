from itertools import product
from astropy.io import fits

from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.swift.swift_filter import SwiftFilter
from swift_comet_pipeline.stacking.stacking import StackingMethod
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.tui.tui_common import stacked_epoch_menu, wait_for_key
from swift_comet_pipeline.pipeline.determine_background import (
    BackgroundDeterminationMethod,
    BackgroundResult,
    background_result_to_dict,
    determine_background,
)
from swift_comet_pipeline.pipeline.pipeline_files import (
    PipelineFiles,
    PipelineProductType,
)


def get_background(img: SwiftUVOTImage, filter_type: SwiftFilter) -> BackgroundResult:
    # TODO: menu here for type of BG method
    bg_cr = determine_background(
        img=img,
        background_method=BackgroundDeterminationMethod.gui_manual_aperture,
        filter_type=filter_type,
    )

    return bg_cr


def background_analysis_step(swift_project_config: SwiftProjectConfig):
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    epoch_id = stacked_epoch_menu(
        pipeline_files=pipeline_files, require_background_analysis_to_be=False
    )
    if epoch_id is None:
        wait_for_key()
        return

    filters = [SwiftFilter.uw1, SwiftFilter.uvv]
    stacking_methods = [StackingMethod.summation, StackingMethod.median]
    for filter_type, stacking_method in product(filters, stacking_methods):
        img_data = pipeline_files.read_pipeline_product(
            PipelineProductType.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        if img_data is None:
            print(
                "Unable to load stacked image of {epoch_id} {filter_to_file_string(filter_type)} {stacking_method}, skipping background analysis ..."
            )
            continue
        img_header = pipeline_files.read_pipeline_product(
            PipelineProductType.stacked_image_header,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )

        bg_results = get_background(img_data, filter_type=filter_type)  # type: ignore

        print(f"Background count rate: {bg_results.count_rate_per_pixel}")

        pipeline_files.write_pipeline_product(
            PipelineProductType.background_analysis,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            data=background_result_to_dict(bg_results),
        )

        bg_corrected_img = img_data - bg_results.count_rate_per_pixel.value

        # make a new fits with the background-corrected image, and copy the header information over from the original stacked image
        bg_hdu = fits.ImageHDU(data=bg_corrected_img)
        bg_hdu.header = img_header

        pipeline_files.write_pipeline_product(
            PipelineProductType.background_subtracted_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            data=bg_hdu,
        )
