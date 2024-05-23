import pathlib
from itertools import product
from typing import Optional

import numpy as np
from icecream import ic

from swift_comet_pipeline.image_manipulation.image_pad import pad_to_match_sizes
from swift_comet_pipeline.observationlog.epoch import (
    Epoch,
    epoch_stacked_image_to_fits,
)

from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.median_divided_image_product import (
    MedianDividedImage,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.median_subtracted_image_product import (
    MedianSubtractedImage,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.q_vs_aperture_radius_product import (
    QvsApertureRadiusProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.radial_profile_image_product import (
    ExtractedRadialProfileImage,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.analysis_step.radial_profile_product import (
    ExtractedRadialProfile,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.background_analysis_step.background_analysis_product import (
    BackgroundAnalysisProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.background_analysis_step.background_subtracted_fits_product import (
    BackgroundSubtractedFITSProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.stacked_epoch_product import (
    StackedEpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.stacked_image_product import (
    StackedFitsImageProduct,
)
from swift_comet_pipeline.stacking.stacked_uvot_image_set import StackedUVOTImageSet
from swift_comet_pipeline.stacking.stacking import stack_epoch_into_sum_and_median
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
    filter_to_string,
)


class EpochSubpipelineFiles:
    """
    For each epoch prepared in the data ingestion step, we have this sub-pipeline to run

    This holds all of the products associated with the sub-pipeline so that we can just pull data out or
    put data in to these products
    """

    def __init__(self, project_path: pathlib.Path, parent_epoch: EpochProduct):
        self.project_path = project_path
        self.parent_epoch = parent_epoch

        self.stacked_epoch = StackedEpochProduct(
            product_path=self.project_path, parent_epoch=self.parent_epoch
        )

        self.stacked_images = {}
        self.background_analyses = {}
        self.background_subtracted_images = {}
        for filter_type, stacking_method in product(
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            self.stacked_images[filter_type, stacking_method] = StackedFitsImageProduct(
                product_path=self.project_path,
                parent_epoch=self.parent_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.background_analyses[filter_type, stacking_method] = (
                BackgroundAnalysisProduct(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

            self.background_subtracted_images[filter_type, stacking_method] = (
                BackgroundSubtractedFITSProduct(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

        self.qh2o_vs_aperture_radius_analyses = {}
        for stacking_method in [StackingMethod.summation, StackingMethod.median]:
            self.qh2o_vs_aperture_radius_analyses[stacking_method] = (
                QvsApertureRadiusProduct(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    stacking_method=stacking_method,
                )
            )

        self.qh2o_from_profile_analyses = {}
        self.extracted_profiles = {}
        self.extracted_profile_images = {}
        self.median_subtracted_images = {}
        self.median_divided_images = {}

        for stacking_method in [StackingMethod.summation, StackingMethod.median]:
            self.qh2o_from_profile_analyses[stacking_method] = QvsApertureRadiusProduct(
                product_path=self.project_path,
                parent_epoch=self.parent_epoch,
                stacking_method=stacking_method,
            )

        for filter_type, stacking_method in product(
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            self.extracted_profiles[filter_type, stacking_method] = (
                ExtractedRadialProfile(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

            self.extracted_profile_images[filter_type, stacking_method] = (
                ExtractedRadialProfileImage(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

            self.median_subtracted_images[filter_type, stacking_method] = (
                MedianSubtractedImage(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

            self.median_divided_images[filter_type, stacking_method] = (
                MedianDividedImage(
                    product_path=self.project_path,
                    parent_epoch=self.parent_epoch,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

    @property
    def all_images_stacked(self) -> bool:
        return all(
            [
                self.stacked_images[f, sm].product_path.exists()
                for f, sm in product(
                    [SwiftFilter.uw1, SwiftFilter.uvv],
                    [StackingMethod.summation, StackingMethod.median],
                )
            ]
        )

    @property
    def background_analyses_done(self) -> bool:
        return all(
            [
                self.background_analyses[f, sm].product_path.exists()
                for f, sm in product(
                    [SwiftFilter.uw1, SwiftFilter.uvv],
                    [StackingMethod.summation, StackingMethod.median],
                )
            ]
        )

    @property
    def background_subtracted_images_generated(self) -> bool:
        return all(
            [
                self.background_subtracted_images[f, sm].product_path.exists()
                for f, sm in product(
                    [SwiftFilter.uw1, SwiftFilter.uvv],
                    [StackingMethod.summation, StackingMethod.median],
                )
            ]
        )

    @property
    def q_vs_aperture_radius_completed(self) -> bool:
        return all(
            [
                self.qh2o_vs_aperture_radius_analyses[sm].product_path.exists()
                for sm in [StackingMethod.summation, StackingMethod.median]
            ]
        )

    def is_stackable(self, epoch: Epoch) -> bool:
        """
        Checks that all uw1 and uvv images in this epoch are taken with the same DATAMODE keyword
        """

        # count the number of unique datamodes: this has to be 1 if we want to stack
        return epoch.DATAMODE.nunique() == 1

    def make_uw1_and_uvv_stacks(
        self,
        swift_data: SwiftData,
        do_coincidence_correction: bool = True,
        remove_vetoed: bool = True,
    ) -> None:
        # TODO: this doesn't belong here, right?
        """
        Produces sum- and median-stacked images for the uw1 and uvv filters
        The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
        """

        uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
        sum_and_median = [StackingMethod.summation, StackingMethod.median]

        # read the parent epoch's observation log
        if self.parent_epoch.data is None:
            self.parent_epoch.read()
        pre_veto_epoch = self.parent_epoch.data
        if pre_veto_epoch is None:
            print(f"Could not read epoch {self.parent_epoch.epoch_id}!")
            return

        # filter out the manually vetoed images from the epoch dataframe?
        if remove_vetoed:
            post_veto_epoch = pre_veto_epoch[pre_veto_epoch.manual_veto == np.False_]
        else:
            post_veto_epoch = pre_veto_epoch

        # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
        if not self.is_stackable(epoch=post_veto_epoch):
            print("Images in the requested stack have mixed data modes! Skipping.")
            return
        else:
            print(
                f"All images taken with FITS keyword DATAMODE={post_veto_epoch.DATAMODE.iloc[0].value}, stacking..."
            )

        # now get just the uw1 and uvv images
        stacked_epoch_mask = np.logical_or(
            post_veto_epoch.FILTER == SwiftFilter.uw1,
            post_veto_epoch.FILTER == SwiftFilter.uvv,
        )
        epoch_to_stack = post_veto_epoch[stacked_epoch_mask]

        # now epoch_to_stack has no vetoed images, and only contains uw1 or uvv images

        epoch_pixel_resolution = epoch_to_stack.ARCSECS_PER_PIXEL.iloc[0]
        stacked_images = StackedUVOTImageSet({})

        # do the stacking
        for filter_type in uw1_and_uvv:
            print(f"Stacking for filter {filter_to_string(filter_type)} ...")

            # now narrow down the data to just one filter at a time
            filter_mask = epoch_to_stack["FILTER"] == filter_type
            epoch_only_this_filter = epoch_to_stack[filter_mask]

            stack_result = stack_epoch_into_sum_and_median(
                swift_data=swift_data,
                epoch=epoch_only_this_filter,
                do_coincidence_correction=do_coincidence_correction,
                pixel_resolution=epoch_pixel_resolution,
            )
            if stack_result is None:
                ic(
                    f"Stacking image for filter {filter_to_file_string(filter_type)} failed!"
                )
                return

            stacked_images[(filter_type, StackingMethod.summation)] = stack_result[0]
            stacked_images[(filter_type, StackingMethod.median)] = stack_result[1]

        # Adjust the images from each filter to be the same size
        for stacking_method in sum_and_median:
            (uw1_img, uvv_img) = pad_to_match_sizes(
                img_one=stacked_images[(SwiftFilter.uw1, stacking_method)],
                img_two=stacked_images[(SwiftFilter.uvv, stacking_method)],
            )
            stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
            stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

        # push all the data into the products for writing later
        self.stacked_epoch.data = epoch_to_stack
        for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
            hdu = epoch_stacked_image_to_fits(
                epoch=epoch_to_stack, img=stacked_images[(filter_type, stacking_method)]
            )
            self.stacked_images[filter_type, stacking_method].data = hdu

    def write_uw1_and_uvv_stacks(self) -> None:
        """
        Writes the stacked epoch dataframe, along with the four images created during stacking
        This is a separate step so that the stacking results can be viewed before deciding to save or not save the results
        """
        uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
        sum_and_median = [StackingMethod.summation, StackingMethod.median]

        self.stacked_epoch.write()

        for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
            self.stacked_images[filter_type, stacking_method].write()

    def get_stacked_image_set(self) -> Optional[StackedUVOTImageSet]:
        stacked_image_set = {}

        uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
        sum_and_median = [StackingMethod.summation, StackingMethod.median]

        # TODO: we should check if any of the data is None and return None if so - we don't have a valid set of stacked images somehow
        for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
            if self.stacked_images[filter_type, stacking_method].data is None:
                self.stacked_images[filter_type, stacking_method].read()

            # the 'data' of the product includes a data.header for the FITS header, and data.data for the numpy image array
            stacked_image_set[filter_type, stacking_method] = self.stacked_images[
                filter_type, stacking_method
            ].data.data

        return stacked_image_set
