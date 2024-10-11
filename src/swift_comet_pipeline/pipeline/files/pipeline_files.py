import calendar
from dataclasses import dataclass
import pathlib
import glob
from itertools import product

import numpy as np
from astropy.time import Time

from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.observationlog.epoch import Epoch, EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import (
    PipelineFilesEnum,
    is_analysis_result_file,
    is_data_ingestion_file,
    is_epoch_subpipeline_file,
)
from swift_comet_pipeline.pipeline.products.data_ingestion.epoch_product import (
    EpochProduct,
)
from swift_comet_pipeline.pipeline.products.data_ingestion.observation_log_product import (
    ObservationLogProduct,
)
from swift_comet_pipeline.pipeline.products.data_ingestion.orbit_product import (
    CometOrbitalDataProduct,
    EarthOrbitalDataProduct,
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
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.exposure_mask_product import (
    ExposureMaskProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.stacked_epoch_product import (
    StackedEpochProduct,
)
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.stacked_image_product import (
    StackedFitsImageProduct,
)
from swift_comet_pipeline.pipeline.products.lightcurve.lightcurve_products import (
    ApertureLightCurveProduct,
    BayesianApertureLightCurveProduct,
    # BayesianVectorialLightCurveProduct,
    BestRednessLightCurveProduct,
    CompleteVectorialLightCurveProduct,
)
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


@dataclass(frozen=True)
class SubpipelineFileKey:
    pf: PipelineFilesEnum
    filter_type: SwiftFilter | None = None
    stacking_method: StackingMethod | None = None


class EpochSubpipelineFiles:
    """
    For each pre-stack epoch prepared in the data ingestion step, we have this sub-pipeline to run

    This holds all of the products associated with this sub-pipeline
    """

    def __init__(
        self, base_project_path: pathlib.Path, parent_pre_stack_epoch: EpochProduct
    ):
        self.base_project_path = base_project_path
        self.parent_pre_stack_epoch = parent_pre_stack_epoch

        self.subpipeline_files = {}
        self.subpipeline_files[
            SubpipelineFileKey(pf=PipelineFilesEnum.epoch_post_stack)
        ] = StackedEpochProduct(
            product_path=self.base_project_path,
            parent_epoch=self.parent_pre_stack_epoch,
        )

        for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
            self.subpipeline_files[
                SubpipelineFileKey(
                    pf=PipelineFilesEnum.exposure_map, filter_type=filter_type
                )
            ] = ExposureMaskProduct(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
            )

        for filter_type, stacking_method in product(
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            self.subpipeline_files[
                SubpipelineFileKey(
                    pf=PipelineFilesEnum.stacked_image,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = StackedFitsImageProduct(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.subpipeline_files[
                SubpipelineFileKey(
                    pf=PipelineFilesEnum.background_determination,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = BackgroundAnalysisProduct(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.subpipeline_files[
                SubpipelineFileKey(
                    pf=PipelineFilesEnum.background_subtracted_image,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = BackgroundSubtractedFITSProduct(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

        for stacking_method in [StackingMethod.summation, StackingMethod.median]:
            self.subpipeline_files[
                SubpipelineFileKey(
                    PipelineFilesEnum.aperture_analysis, stacking_method=stacking_method
                )
            ] = QvsApertureRadiusProduct(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                stacking_method=stacking_method,
            )

        for filter_type, stacking_method in product(
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            self.subpipeline_files[
                SubpipelineFileKey(
                    PipelineFilesEnum.extracted_profile,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = ExtractedRadialProfile(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.subpipeline_files[
                SubpipelineFileKey(
                    PipelineFilesEnum.extracted_profile_image,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = ExtractedRadialProfileImage(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.subpipeline_files[
                SubpipelineFileKey(
                    PipelineFilesEnum.median_subtracted_image,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = MedianSubtractedImage(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

            self.subpipeline_files[
                SubpipelineFileKey(
                    PipelineFilesEnum.median_divided_image,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            ] = MedianDividedImage(
                product_path=self.base_project_path,
                parent_epoch=self.parent_pre_stack_epoch,
                filter_type=filter_type,
                stacking_method=stacking_method,
            )

    def get_product(self, key: SubpipelineFileKey):
        return self.subpipeline_files[key]

    # def make_uw1_and_uvv_stacks(
    #     self,
    #     swift_data: SwiftData,
    #     do_coincidence_correction: bool = True,
    #     remove_vetoed: bool = True,
    # ) -> None:
    #     # TODO: this doesn't belong here, right?
    #     """
    #     Produces sum- and median-stacked images for the uw1 and uvv filters
    #     The stacked images are padded so that the images in uw1 and uvv are the same size, so both must be stacked here
    #     """
    #
    #     uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    #     sum_and_median = [StackingMethod.summation, StackingMethod.median]
    #
    #     # read the parent epoch's observation log
    #     if self.parent_epoch.data is None:
    #         self.parent_epoch.read()
    #     pre_veto_epoch = self.parent_epoch.data
    #     if pre_veto_epoch is None:
    #         print(f"Could not read epoch {self.parent_epoch.epoch_id}!")
    #         return
    #
    #     # filter out the manually vetoed images from the epoch dataframe?
    #     if remove_vetoed:
    #         post_veto_epoch = pre_veto_epoch[pre_veto_epoch.manual_veto == np.False_]
    #     else:
    #         post_veto_epoch = pre_veto_epoch
    #
    #     # are we stacking images with mixed data modes (and therefore mixed pixel resolutions?)
    #     if not self.is_stackable(epoch=post_veto_epoch):
    #         print("Images in the requested stack have mixed data modes! Skipping.")
    #         return
    #     else:
    #         print(
    #             f"All images taken with FITS keyword DATAMODE={post_veto_epoch.DATAMODE.iloc[0].value}, stacking..."
    #         )
    #
    #     # now get just the uw1 and uvv images
    #     stacked_epoch_mask = np.logical_or(
    #         post_veto_epoch.FILTER == SwiftFilter.uw1,
    #         post_veto_epoch.FILTER == SwiftFilter.uvv,
    #     )
    #     epoch_to_stack = post_veto_epoch[stacked_epoch_mask]
    #
    #     # now epoch_to_stack has no vetoed images, and only contains uw1 or uvv images
    #
    #     epoch_pixel_resolution = epoch_to_stack.ARCSECS_PER_PIXEL.iloc[0]
    #     stacked_images = StackedUVOTImageSet({})
    #     exposure_maps = {}
    #
    #     # do the stacking
    #     for filter_type in uw1_and_uvv:
    #         print(f"Stacking for filter {filter_to_string(filter_type)} ...")
    #
    #         # now narrow down the data to just one filter at a time
    #         filter_mask = epoch_to_stack["FILTER"] == filter_type
    #         epoch_only_this_filter = epoch_to_stack[filter_mask]
    #
    #         stack_result = stack_epoch_into_sum_and_median(
    #             swift_data=swift_data,
    #             epoch=epoch_only_this_filter,
    #             do_coincidence_correction=do_coincidence_correction,
    #             pixel_resolution=epoch_pixel_resolution,
    #         )
    #         if stack_result is None:
    #             ic(
    #                 f"Stacking image for filter {filter_to_file_string(filter_type)} failed!"
    #             )
    #             return
    #
    #         stacked_images[(filter_type, StackingMethod.summation)] = stack_result[0]
    #         stacked_images[(filter_type, StackingMethod.median)] = stack_result[1]
    #         exposure_maps[filter_type] = stack_result[2]
    #
    #     # Adjust the images from each filter to be the same size
    #     for stacking_method in sum_and_median:
    #         (uw1_img, uvv_img) = pad_to_match_sizes(
    #             img_one=stacked_images[(SwiftFilter.uw1, stacking_method)],
    #             img_two=stacked_images[(SwiftFilter.uvv, stacking_method)],
    #         )
    #         stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
    #         stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img
    #
    #     # Adjust the exposure maps as well so that they stay the same size as the stacked images
    #     uw1_exp_map, uvv_exp_map = pad_to_match_sizes(
    #         img_one=exposure_maps[SwiftFilter.uw1],
    #         img_two=exposure_maps[SwiftFilter.uvv],
    #     )
    #
    #     # push all the data into the products for writing later
    #     self.stacked_epoch.data = epoch_to_stack
    #     for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
    #         hdu = epoch_stacked_image_to_fits(
    #             epoch=epoch_to_stack, img=stacked_images[(filter_type, stacking_method)]
    #         )
    #         self.stacked_images[filter_type, stacking_method].data = hdu
    #
    #     self.exposure_map[SwiftFilter.uw1].data = epoch_stacked_image_to_fits(
    #         epoch=epoch_to_stack, img=uw1_exp_map
    #     )
    #     self.exposure_map[SwiftFilter.uvv].data = epoch_stacked_image_to_fits(
    #         epoch=epoch_to_stack, img=uvv_exp_map
    #     )
    #
    # def write_uw1_and_uvv_stacks(self) -> None:
    #     """
    #     Writes the stacked epoch dataframe, along with the four images created during stacking, and exposure map
    #     This is a separate step so that the stacking results can be viewed before deciding to save or not save the results
    #     """
    #     uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    #     sum_and_median = [StackingMethod.summation, StackingMethod.median]
    #
    #     self.stacked_epoch.write()
    #
    #     for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
    #         self.stacked_images[filter_type, stacking_method].write()
    #
    #     for filter_type in [SwiftFilter.uw1, SwiftFilter.uvv]:
    #         self.exposure_map[filter_type].write()
    #
    # def get_stacked_image_set(self) -> StackedUVOTImageSet | None:
    #     stacked_image_set = {}
    #
    #     uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
    #     sum_and_median = [StackingMethod.summation, StackingMethod.median]
    #
    #     # TODO: we should check if any of the data is None and return None if so - we don't have a valid set of stacked images somehow
    #     for filter_type, stacking_method in product(uw1_and_uvv, sum_and_median):
    #         if self.stacked_images[filter_type, stacking_method].data is None:
    #             self.stacked_images[filter_type, stacking_method].read()
    #
    #         # the 'data' of the product includes a data.header for the FITS header, and data.data for the numpy image array
    #         stacked_image_set[filter_type, stacking_method] = self.stacked_images[
    #             filter_type, stacking_method
    #         ].data.data
    #
    #     return stacked_image_set


@dataclass(frozen=True)
class AnalysisFileKey:
    pf: PipelineFilesEnum
    stacking_method: StackingMethod
    # fit_type: VectorialFitType | None = None


class AnalysisFiles:

    def __init__(self, base_project_path: pathlib.Path):
        self.base_project_path = base_project_path

        self.analysis_products = {}
        for stacking_method in [StackingMethod.summation, StackingMethod.median]:
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.aperture_lightcurve,
                    stacking_method=stacking_method,
                )
            ] = ApertureLightCurveProduct(
                product_path=self.base_project_path, stacking_method=stacking_method
            )
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.bayesian_aperture_lightcurve,
                    stacking_method=stacking_method,
                )
            ] = BayesianApertureLightCurveProduct(
                product_path=self.base_project_path, stacking_method=stacking_method
            )
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.complete_vectorial_lightcurve,
                    stacking_method=stacking_method,
                )
            ] = CompleteVectorialLightCurveProduct(
                product_path=self.base_project_path, stacking_method=stacking_method
            )
            # self.analysis_products[
            #     AnalysisFileKey(
            #         PipelineFilesEnum.bayesian_vectorial_lightcurve,
            #         stacking_method=stacking_method,
            #     )
            # ] = BayesianVectorialLightCurveProduct(
            #     product_path=self.base_project_path, stacking_method=stacking_method
            # )
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
                    stacking_method=stacking_method,
                    # fit_type=VectorialFitType.near_fit,
                )
            ] = BestRednessLightCurveProduct(
                product_path=self.base_project_path,
                stacking_method=stacking_method,
                fit_type=VectorialFitType.near_fit,
            )
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
                    stacking_method=stacking_method,
                    # fit_type=VectorialFitType.far_fit,
                )
            ] = BestRednessLightCurveProduct(
                product_path=self.base_project_path,
                stacking_method=stacking_method,
                fit_type=VectorialFitType.far_fit,
            )
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
                    stacking_method=stacking_method,
                    # fit_type=VectorialFitType.full_fit,
                )
            ] = BestRednessLightCurveProduct(
                product_path=self.base_project_path,
                stacking_method=stacking_method,
                fit_type=VectorialFitType.full_fit,
            )

    def get_product(self, key: AnalysisFileKey):
        return self.analysis_products[key]


class SwiftCometPipelineFiles:

    def __init__(self, base_project_path: pathlib.Path):
        self.base_project_path = base_project_path

        self._build_data_ingestion_files()

        self.pre_stack_epochs_path = self.base_project_path / pathlib.Path("epochs")
        self.pre_stack_epochs_path.mkdir(exist_ok=True)
        self.pre_stack_epochs: list[EpochProduct] | None = None
        self.epoch_ids_to_epoch_products: dict[EpochID, EpochProduct] | None = None
        self._scan_for_pre_stack_epoch_files()
        self._build_epoch_subpipelines()
        self.analysis_products = AnalysisFiles(base_project_path=self.base_project_path)

    def _build_data_ingestion_files(self):
        """
        Modifies: self.data_ingestion_products
        """
        self.data_ingestion_products = {}
        self.data_ingestion_products[PipelineFilesEnum.observation_log] = (
            ObservationLogProduct(product_path=self.base_project_path)
        )
        self.data_ingestion_products[PipelineFilesEnum.comet_orbital_data] = (
            CometOrbitalDataProduct(product_path=self.base_project_path)
        )
        self.data_ingestion_products[PipelineFilesEnum.earth_orbital_data] = (
            EarthOrbitalDataProduct(product_path=self.base_project_path)
        )

    def _find_epoch_files(self) -> list[pathlib.Path] | None:
        """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
        glob_pattern = str(self.pre_stack_epochs_path / pathlib.Path("*.parquet"))
        epoch_filename_list = sorted(glob.glob(glob_pattern))
        if len(epoch_filename_list) == 0:
            return None
        return [pathlib.Path(x) for x in epoch_filename_list]

    def _scan_for_pre_stack_epoch_files(self):
        """
        Modifies: self.pre_stack_epochs, self.epoch_ids_to_epoch_products

        Look on disk for existing epochs and create EpochProducts out of them if they are there
        """
        epoch_path_list = self._find_epoch_files()

        if epoch_path_list is None:
            self.pre_stack_epochs = None
            self.epoch_ids_to_epoch_products = None
            return

        self.pre_stack_epochs = [EpochProduct(product_path=x) for x in epoch_path_list]

        self.epoch_ids_to_epoch_products = {}
        for parent_pre_stack_epoch in self.pre_stack_epochs:
            self.epoch_ids_to_epoch_products[parent_pre_stack_epoch.epoch_id] = (
                parent_pre_stack_epoch
            )

    def create_pre_stack_epochs(
        self, epoch_list: list[Epoch], write_to_disk: bool
    ) -> None:
        """
        Modifies: self.pre_stack_epochs

        The epochs should be time-sorted before they are passed in!
        """

        # generate a file path for each each programmatically
        epoch_path_list = []
        for i, epoch in enumerate(epoch_list):
            epoch_mid = Time(np.min(epoch.MID_TIME)).ymdhms
            day = epoch_mid.day  # type: ignore
            month = calendar.month_abbr[epoch_mid.month]  # type: ignore
            year = epoch_mid.year  # type: ignore

            epoch_path_list.append(
                pathlib.Path(f"{i:03d}_{year}_{day:02d}_{month}.parquet")
            )

        # create the product objects with the generated paths
        epoch_products = [
            EpochProduct(product_path=self.pre_stack_epochs_path / x)
            for x in epoch_path_list
        ]

        # put the epoch dataframes into the product to be written with the proper output format
        for epoch_product, epoch in zip(epoch_products, epoch_list):
            epoch_product.data = epoch
            if write_to_disk:
                epoch_product.write()

        # update
        self._scan_for_pre_stack_epoch_files()

    # def delete_pre_stack_epochs(self) -> None:
    #     if not self.pre_stack_epochs:
    #         return
    #
    #     for epoch_product in self.pre_stack_epochs:
    #         epoch_product.delete_file()
    #
    #     self.pre_st = None

    def _build_epoch_subpipelines(self):
        """
        Modifies: self.epoch_subpipelines
        """

        if self.pre_stack_epochs is None:
            self.epoch_subpipelines = None
            return

        self.epoch_subpipelines = {}
        for parent_pre_stack_epoch in self.pre_stack_epochs:
            self.epoch_subpipelines[parent_pre_stack_epoch] = EpochSubpipelineFiles(
                base_project_path=self.base_project_path,
                parent_pre_stack_epoch=parent_pre_stack_epoch,
            )

    def get_product_with_key(
        self,
        pf: PipelineFilesEnum,
        parent_pre_stack_epoch: EpochProduct | None = None,
        key: SubpipelineFileKey | AnalysisFileKey | None = None,
    ) -> PipelineProduct | None:

        if is_data_ingestion_file(pf):
            return self.data_ingestion_products[pf]

        if is_analysis_result_file(pf):
            if not isinstance(key, AnalysisFileKey):
                return None
            return self.analysis_products.get_product(key)

        if is_epoch_subpipeline_file(pf):
            if (
                not isinstance(key, SubpipelineFileKey)
                or self.epoch_subpipelines is None
            ):
                return None
            return self.epoch_subpipelines[parent_pre_stack_epoch].get_product(key)

    def get_product(
        self,
        pf: PipelineFilesEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        # fit_type: VectorialFitType | None = None,
    ) -> PipelineProduct | None:

        if is_data_ingestion_file(pf):
            return self.data_ingestion_products[pf]

        if is_analysis_result_file(pf):
            if stacking_method is None:
                return None
            return self.analysis_products.get_product(
                AnalysisFileKey(pf=pf, stacking_method=stacking_method)
                # AnalysisFileKey(
                #     pf=pf, stacking_method=stacking_method, fit_type=fit_type
                # )
            )

        if is_epoch_subpipeline_file(pf):
            if (
                self.epoch_subpipelines is None
                or self.epoch_ids_to_epoch_products is None
                or epoch_id is None
            ):
                return None
            parent_pre_stack_epoch = self.epoch_ids_to_epoch_products.get(
                epoch_id, None
            )
            if pf == PipelineFilesEnum.epoch_pre_stack:
                return parent_pre_stack_epoch
            if parent_pre_stack_epoch is None:
                return None
            key = SubpipelineFileKey(
                pf=pf, filter_type=filter_type, stacking_method=stacking_method
            )
            return self.epoch_subpipelines[parent_pre_stack_epoch].get_product(key)

    def get_epoch_id_list(self) -> list[EpochID] | None:
        if self.pre_stack_epochs is None or self.epoch_ids_to_epoch_products is None:
            return None

        return list(self.epoch_ids_to_epoch_products.keys())

    def exists(
        self,
        pf: PipelineFilesEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        # fit_type: VectorialFitType | None = None,
    ) -> bool:

        p = self.get_product(
            pf=pf,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            # fit_type=fit_type,
        )
        if p is None:
            return False
        return p.exists()
