from astropy.io import fits
from swift_comet_pipeline.configs import SwiftProjectConfig

from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.tui import stacked_epoch_menu
from swift_comet_pipeline.determine_background import (
    BackgroundDeterminationMethod,
    BackgroundResult,
    background_analysis_to_yaml_dict,
    determine_background,
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

    epoch_path = stacked_epoch_menu(pipeline_files=pipeline_files)
    if epoch_path is None:
        print("No stacked images found! Exiting.")
        return 1
    if pipeline_files.stacked_epoch_products is None:
        print(
            "Pipeline error! This is a bug with pipeline_files.stacked_epoch_products!"
        )
        return 1

    # stacking_method = StackingMethod.summation
    for stacking_method in [StackingMethod.summation, StackingMethod.median]:
        if pipeline_files.stacked_image_products is None:
            print(
                "Pipeline error! This is a bug with pipeline_files.stacked_image_products!"
            )
            return 1
        uw1_sum_prod = pipeline_files.stacked_image_products[
            epoch_path, SwiftFilter.uw1, stacking_method
        ]
        uw1_sum_prod.load_product()
        uw1_sum = uw1_sum_prod.data_product.data

        uvv_sum_prod = pipeline_files.stacked_image_products[
            epoch_path, SwiftFilter.uvv, stacking_method
        ]
        uvv_sum_prod.load_product()
        uvv_sum = uvv_sum_prod.data_product.data

        # do the background analysis
        bguw1_results = get_background(uw1_sum, filter_type=SwiftFilter.uw1)
        bguvv_results = get_background(uvv_sum, filter_type=SwiftFilter.uvv)

        # extract the count rates
        bguw1 = bguw1_results.count_rate_per_pixel
        bguvv = bguvv_results.count_rate_per_pixel
        print("")
        print(f"Background count rate for uw1: {bguw1}")
        print(f"Background count rate for uvv: {bguvv}")

        uw1 = uw1_sum - bguw1.value
        uvv = uvv_sum - bguvv.value

        # save the results from the analysis, including the background-subtracted images
        if pipeline_files.analysis_background_products is None:
            print(
                "Pipeline error! This is a bug with pipeline_files.analysis_background_products!"
            )
            return 1
        bg_analysis_prod = pipeline_files.analysis_background_products[
            epoch_path, stacking_method
        ]
        bg_analysis_prod.data_product = background_analysis_to_yaml_dict(
            method=BackgroundDeterminationMethod.gui_manual_aperture,
            uw1_result=bguw1_results,
            uvv_result=bguvv_results,
        )
        bg_analysis_prod.product_path.parent.mkdir(parents=True, exist_ok=True)
        bg_analysis_prod.save_product()

        if pipeline_files.analysis_bg_subtracted_images is None:
            print(
                "Pipeline error! This is a bug with pipeline_files.analysis_bg_subtracted_images!"
            )
            return 1

        bg_images = {SwiftFilter.uw1: uw1, SwiftFilter.uvv: uvv}
        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            bg_sub_image_prod = pipeline_files.analysis_bg_subtracted_images[
                epoch_path, filter_type, stacking_method
            ]

            stacked_image_prod = pipeline_files.stacked_image_products[
                epoch_path, filter_type, stacking_method
            ]
            # grab the original stacked image header and copy to background-subtracted fits
            stacked_hdul = fits.open(stacked_image_prod.product_path)
            hdr_to_copy = stacked_hdul[1].header  # type: ignore

            bg_hdu = fits.ImageHDU(data=bg_images[filter_type])
            bg_hdu.header = hdr_to_copy

            bg_sub_image_prod.data_product = bg_hdu
            bg_sub_image_prod.save_product()

            stacked_hdul.close()
