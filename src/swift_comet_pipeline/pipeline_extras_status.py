from rich import print as rprint
from rich.panel import Panel

from swift_comet_pipeline.configs import SwiftProjectConfig
from swift_comet_pipeline.pipeline_files import PipelineFiles
from swift_comet_pipeline.stacking import StackingMethod
from swift_comet_pipeline.swift_filter import SwiftFilter
from swift_comet_pipeline.tui import bool_to_x_or_check


def pipeline_extra_status(swift_project_config: SwiftProjectConfig) -> None:
    pipeline_files = PipelineFiles(swift_project_config.product_save_path)

    print("")

    rprint(
        "[orange3]Observation log:[/orange3] ",
        bool_to_x_or_check(pipeline_files.observation_log.exists()),
    )
    rprint(
        "[orange3]Comet orbit data:[/orange3] ",
        # pipeline_files.comet_orbital_data.product_path,
        bool_to_x_or_check(pipeline_files.comet_orbital_data.exists()),
    )
    rprint(
        "[orange3]Earth orbit data:[/orange3] ",
        # pipeline_files.earth_orbital_data.product_path,
        bool_to_x_or_check(pipeline_files.earth_orbital_data.exists()),
    )

    print("")

    if pipeline_files.epoch_products is None:
        print("No epochs defined yet!")
        return

    print("Epochs:")
    for x in pipeline_files.epoch_products:
        print(x.product_path.stem)

    print("")

    print("Stacked epochs and images:")
    if pipeline_files.stacked_epoch_products is None:
        print("No stacked products found!")
        return

    for epoch_prod in pipeline_files.epoch_products:
        epoch_path = epoch_prod.product_path

        # # print whether this particular epoch has been stacked by looking for its stacked epoch
        # ep_prod = pipeline_files.stacked_epoch_products[epoch_path]
        # print(ep_prod.product_path, bool_to_x_or_check(epoch_path.exists()))

        rprint(Panel(f"Epoch {epoch_path.stem}:", expand=False))
        # does the stacked image associated with this stacked epoch exist?
        if pipeline_files.stacked_image_products is not None:
            uw1_sum = pipeline_files.stacked_image_products[
                epoch_path, SwiftFilter.uw1, StackingMethod.summation
            ]
            uw1_median = pipeline_files.stacked_image_products[
                epoch_path, SwiftFilter.uw1, StackingMethod.median
            ]
            uvv_sum = pipeline_files.stacked_image_products[
                epoch_path, SwiftFilter.uvv, StackingMethod.summation
            ]
            uvv_median = pipeline_files.stacked_image_products[
                epoch_path, SwiftFilter.uvv, StackingMethod.median
            ]
            rprint(
                "Stacked images: ",
                f"uw1 sum: {bool_to_x_or_check(uw1_sum.exists())}\t",
                f"uw1 median: {bool_to_x_or_check(uw1_median.exists())}\t",
                f"uvv sum: {bool_to_x_or_check(uvv_sum.exists())}\t",
                f"uvv median: {bool_to_x_or_check(uvv_median.exists())}",
            )

        if pipeline_files.analysis_background_products is not None:
            bg_prod_sum = pipeline_files.analysis_background_products[
                epoch_path, StackingMethod.summation
            ]
            bg_prod_median = pipeline_files.analysis_background_products[
                epoch_path, StackingMethod.median
            ]
            sum_method = ""
            median_method = ""
            if bg_prod_sum.exists():
                bg_prod_sum.load_product()
                # print(bg_prod_sum.data_product["method"])
                sum_method = f" ({str(bg_prod_sum.data_product['method'])})"
            if bg_prod_median.exists():
                bg_prod_median.load_product()
                median_method = f" ({str(bg_prod_median.data_product['method'])})"
            rprint(
                "Background analysis: ",
                f"sum: {bool_to_x_or_check(bg_prod_sum.exists())}{sum_method} ",
                f"median: {bool_to_x_or_check(bg_prod_median.exists())}{median_method}",
            )

        if pipeline_files.analysis_bg_subtracted_images is not None:
            uw1_bg_sub = pipeline_files.analysis_bg_subtracted_images[
                epoch_path, SwiftFilter.uw1, StackingMethod.summation
            ]
            uvv_bg_sub = pipeline_files.analysis_bg_subtracted_images[
                epoch_path, SwiftFilter.uvv, StackingMethod.summation
            ]
            rprint(
                "Background-subtracted images: ",
                "uw1: ",
                # uw1_bg_sub.product_path.name,
                bool_to_x_or_check(uw1_bg_sub.exists()),
                "uvv: ",
                # uvv_bg_sub.product_path.name,
                bool_to_x_or_check(uvv_bg_sub.exists()),
            )
            # if uw1_bg_sub.exists():
            #     uw1_bg_sub.load_product()
            #     print("Dimensions:", uw1_bg_sub.data_product.data.shape)
            # if uvv_bg_sub.exists():
            #     uvv_bg_sub.load_product()
            #     print("Dimensions:", uvv_bg_sub.data_product.data.shape)

        if pipeline_files.analysis_qh2o_products is not None:
            q = pipeline_files.analysis_qh2o_products[epoch_path]
            rprint(
                "Q vs aperture radius analysis: ",
                # q.product_path.name,
                bool_to_x_or_check(q.exists()),
            )
            # if q.exists():
            #     q.load_product()
            #     print(q.data_product.Q_H2O[0])

        print("")
