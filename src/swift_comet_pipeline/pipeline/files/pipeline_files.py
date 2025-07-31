import calendar
from dataclasses import dataclass
import pathlib
import glob
from itertools import product

import numpy as np
from astropy.time import Time

from swift_comet_pipeline.observationlog.epoch_typing import Epoch, EpochID
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
from swift_comet_pipeline.pipeline.products.epoch_subpipeline.stacking_step.exposure_map_product import (
    ExposureMapProduct,
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
    UnifiedLightCurveProduct,
)
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.types.stacking_method import StackingMethod
from swift_comet_pipeline.types.swift_filter import SwiftFilter
from swift_comet_pipeline.types.vectorial_model_fit_type import VectorialFitType


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
            ] = ExposureMapProduct(
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


@dataclass(frozen=True)
class AnalysisFileKey:
    pf: PipelineFilesEnum
    stacking_method: StackingMethod


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
            self.analysis_products[
                AnalysisFileKey(
                    PipelineFilesEnum.unified_lightcurve,
                    stacking_method=stacking_method,
                )
            ] = UnifiedLightCurveProduct(
                product_path=self.base_project_path, stacking_method=stacking_method
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
