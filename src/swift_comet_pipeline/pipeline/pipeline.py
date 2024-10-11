import functools
from itertools import product
from typing import Any

from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.observationlog.epoch import EpochID
from swift_comet_pipeline.pipeline.files.pipeline_files import SwiftCometPipelineFiles
from swift_comet_pipeline.pipeline.files.pipeline_files_enum import PipelineFilesEnum
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.pipeline.steps.pipeline_steps import (
    ApertureAnalysisStep,
    BackgroundSubtractStep,
    BuildLightcurvesStep,
    DetermineBackgroundStep,
    DownloadOrbitalDataStep,
    EpochStackStep,
    IdentifyEpochsStep,
    ObservationLogStep,
    SwiftCometPipelineStepStatus,
    VectorialAnalysisStep,
    VetoImagesStep,
)
from swift_comet_pipeline.pipeline.steps.pipeline_steps_enum import (
    SwiftCometPipelineStepEnum,
)
from swift_comet_pipeline.projects.configs import SwiftProjectConfig
from swift_comet_pipeline.stacking.stacking_method import StackingMethod
from swift_comet_pipeline.swift.swift_filter import SwiftFilter


class SwiftCometPipeline:

    def __init__(self, swift_project_config: SwiftProjectConfig):
        self.spc = swift_project_config
        self._construct_pipeline_steps()
        self._refresh_pipeline_file_list()

        self.uw1_and_uvv = [SwiftFilter.uw1, SwiftFilter.uvv]
        self.sum_and_median = [StackingMethod.summation, StackingMethod.median]

    def _construct_pipeline_steps(self):
        self.steps = {}
        self.steps[SwiftCometPipelineStepEnum.observation_log] = ObservationLogStep()
        self.steps[SwiftCometPipelineStepEnum.download_orbital_data] = (
            DownloadOrbitalDataStep()
        )
        self.steps[SwiftCometPipelineStepEnum.identify_epochs] = IdentifyEpochsStep()
        self.steps[SwiftCometPipelineStepEnum.veto_images] = VetoImagesStep()
        self.steps[SwiftCometPipelineStepEnum.epoch_stack] = EpochStackStep()
        self.steps[SwiftCometPipelineStepEnum.determine_background] = (
            DetermineBackgroundStep()
        )
        self.steps[SwiftCometPipelineStepEnum.background_subtract] = (
            BackgroundSubtractStep()
        )
        self.steps[SwiftCometPipelineStepEnum.aperture_analysis] = (
            ApertureAnalysisStep()
        )
        self.steps[SwiftCometPipelineStepEnum.vectorial_analysis] = (
            VectorialAnalysisStep()
        )
        self.steps[SwiftCometPipelineStepEnum.build_lightcurves] = (
            BuildLightcurvesStep()
        )

    def _refresh_pipeline_file_list(self):
        self.pipeline_files = SwiftCometPipelineFiles(
            base_project_path=self.spc.project_path
        )

    def _bool_to_status(self, x):
        if x:
            return SwiftCometPipelineStepStatus.complete
        else:
            return SwiftCometPipelineStepStatus.not_complete

    def get_status(
        self,
        step: SwiftCometPipelineStepEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
    ) -> SwiftCometPipelineStepStatus:

        match step:
            case SwiftCometPipelineStepEnum.observation_log:
                return self._get_observation_log_status()
            case SwiftCometPipelineStepEnum.download_orbital_data:
                return self._get_orbital_data_status()
            case SwiftCometPipelineStepEnum.identify_epochs:
                return self._get_identify_epoch_status()
            case SwiftCometPipelineStepEnum.veto_images:
                return self._get_veto_images_status()
            case SwiftCometPipelineStepEnum.epoch_stack:
                return self._get_epoch_stack_status(
                    epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case SwiftCometPipelineStepEnum.determine_background:
                return self._get_determine_background_status(
                    epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case SwiftCometPipelineStepEnum.background_subtract:
                return self._get_background_subtract_status(
                    epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case SwiftCometPipelineStepEnum.aperture_analysis:
                return self._get_aperture_analysis_status(
                    epoch_id=epoch_id,
                    # filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case SwiftCometPipelineStepEnum.vectorial_analysis:
                return self._get_vectorial_analysis_status(
                    epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case SwiftCometPipelineStepEnum.build_lightcurves:
                return self._get_build_lightcurve_status(
                    epoch_id=epoch_id,
                    stacking_method=stacking_method,
                )

    def _status_list_to_single_status(
        self, stats: list[SwiftCometPipelineStepStatus]
    ) -> SwiftCometPipelineStepStatus:
        """
        If any in the list is 'invalid', the return value is 'invalid'.
        If all are 'complete', the return value is 'complete'.
        If all are 'not_complate', the return value is 'not_complete'.
        Otherwise, the return value is 'partial'.
        """

        if SwiftCometPipelineStepStatus.invalid in stats:
            return SwiftCometPipelineStepStatus.invalid

        stat_complete = [x == SwiftCometPipelineStepStatus.complete for x in stats]
        if all(stat_complete):
            return SwiftCometPipelineStepStatus.complete

        stat_incomplete = [
            x == SwiftCometPipelineStepStatus.not_complete for x in stats
        ]
        if all(stat_incomplete):
            return SwiftCometPipelineStepStatus.not_complete

        return SwiftCometPipelineStepStatus.partial

    def get_overall_status(
        self, step: SwiftCometPipelineStepEnum
    ) -> SwiftCometPipelineStepStatus:

        match step:
            case SwiftCometPipelineStepEnum.observation_log:
                return self._get_observation_log_status()
            case SwiftCometPipelineStepEnum.download_orbital_data:
                return self._get_orbital_data_status()
            case SwiftCometPipelineStepEnum.identify_epochs:
                return self._get_identify_epoch_status()
            case SwiftCometPipelineStepEnum.veto_images:
                return self._get_veto_images_status()
            case SwiftCometPipelineStepEnum.epoch_stack:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_epoch_stack_status(e, f, s)
                    for e, f, s in product(
                        epochs, self.uw1_and_uvv, self.sum_and_median
                    )
                ]
                return self._status_list_to_single_status(stats)
            case SwiftCometPipelineStepEnum.determine_background:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_determine_background_status(e, f, s)
                    for e, f, s in product(
                        epochs, self.uw1_and_uvv, self.sum_and_median
                    )
                ]
                return self._status_list_to_single_status(stats)
            case SwiftCometPipelineStepEnum.background_subtract:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_background_subtract_status(e, f, s)
                    for e, f, s in product(
                        epochs, self.uw1_and_uvv, self.sum_and_median
                    )
                ]
                return self._status_list_to_single_status(stats)
            case SwiftCometPipelineStepEnum.aperture_analysis:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_aperture_analysis_status(e, s)
                    for e, s in product(epochs, self.sum_and_median)
                ]
                return self._status_list_to_single_status(stats)
            case SwiftCometPipelineStepEnum.vectorial_analysis:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_vectorial_analysis_status(e, f, s)
                    for e, f, s in product(
                        epochs, self.uw1_and_uvv, self.sum_and_median
                    )
                ]
                return self._status_list_to_single_status(stats)
            case SwiftCometPipelineStepEnum.build_lightcurves:
                epochs = self.get_epoch_id_list()
                if epochs is None:
                    return SwiftCometPipelineStepStatus.not_complete
                stats = [
                    self._get_build_lightcurve_status(e, s)
                    for e, s in product(epochs, self.sum_and_median)
                ]
                return self._status_list_to_single_status(stats)

    def _get_observation_log_status(self) -> SwiftCometPipelineStepStatus:
        obs_log_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.observation_log
        )
        return self._bool_to_status(obs_log_exists)

    def _get_orbital_data_status(self) -> SwiftCometPipelineStepStatus:
        comet_orb_data = self.pipeline_files.exists(
            pf=PipelineFilesEnum.comet_orbital_data
        )
        earth_orb_data = self.pipeline_files.exists(
            pf=PipelineFilesEnum.earth_orbital_data
        )
        if comet_orb_data and earth_orb_data:
            return SwiftCometPipelineStepStatus.complete
        if not comet_orb_data and not earth_orb_data:
            return SwiftCometPipelineStepStatus.not_complete
        else:
            return SwiftCometPipelineStepStatus.partial

    def _get_identify_epoch_status(self) -> SwiftCometPipelineStepStatus:
        if self.pipeline_files.pre_stack_epochs is None:
            return SwiftCometPipelineStepStatus.not_complete
        else:
            return SwiftCometPipelineStepStatus.complete

    def _get_veto_images_status(self) -> SwiftCometPipelineStepStatus:
        # we don't have a way if knowing if the images were reviewed and vetoed - so link this to the fact that the pre-stack epochs exist
        return self._get_identify_epoch_status()

    def _get_determine_background_status(
        self,
        epoch_id: EpochID | None,
        filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or filter_type is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        bg_det_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.background_determination,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        return self._bool_to_status(bg_det_exists)

    def _get_background_subtract_status(
        self,
        epoch_id: EpochID | None,
        filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or filter_type is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        bg_img_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.background_subtracted_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        return self._bool_to_status(bg_img_exists)

    def _get_epoch_stack_status(
        self,
        epoch_id: EpochID | None,
        filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or filter_type is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        stacked_epoch_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.epoch_post_stack,
            epoch_id=epoch_id,
        )
        stack_img_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.stacked_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        exp_map_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.exposure_map,
            epoch_id=epoch_id,
            filter_type=filter_type,
        )
        if stacked_epoch_exists and stack_img_exists and exp_map_exists:
            return SwiftCometPipelineStepStatus.complete
        if not stacked_epoch_exists and not stack_img_exists and not exp_map_exists:
            return SwiftCometPipelineStepStatus.not_complete
        return SwiftCometPipelineStepStatus.partial

    def _get_aperture_analysis_status(
        self,
        epoch_id: EpochID | None,
        # filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        ap_analysis_exists = self.pipeline_files.exists(
            pf=PipelineFilesEnum.aperture_analysis,
            epoch_id=epoch_id,
            stacking_method=stacking_method,
        )
        return self._bool_to_status(ap_analysis_exists)

    def _get_vectorial_analysis_status(
        self,
        epoch_id: EpochID | None,
        filter_type: SwiftFilter | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or filter_type is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        extracted_prof = self.pipeline_files.exists(
            pf=PipelineFilesEnum.extracted_profile,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        extracted_prof_img = self.pipeline_files.exists(
            pf=PipelineFilesEnum.extracted_profile_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        med_sub_img = self.pipeline_files.exists(
            pf=PipelineFilesEnum.median_subtracted_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        med_div_img = self.pipeline_files.exists(
            pf=PipelineFilesEnum.median_divided_image,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        products_exist = [extracted_prof, extracted_prof_img, med_sub_img, med_div_img]
        # print(f"{epoch_id=}\t{filter_type=}\t{stacking_method=}\t{products_exist=}")
        if all(products_exist):
            return SwiftCometPipelineStepStatus.complete
        if all([not x for x in products_exist]):
            return SwiftCometPipelineStepStatus.not_complete
        return SwiftCometPipelineStepStatus.partial

    def _get_build_lightcurve_status(
        self,
        epoch_id: EpochID | None,
        stacking_method: StackingMethod | None,
    ) -> SwiftCometPipelineStepStatus:
        if epoch_id is None or stacking_method is None:
            return SwiftCometPipelineStepStatus.invalid
        products = [
            PipelineFilesEnum.aperture_lightcurve,
            PipelineFilesEnum.bayesian_aperture_lightcurve,
            PipelineFilesEnum.complete_vectorial_lightcurve,
            # PipelineFilesEnum.bayesian_vectorial_lightcurve,
            PipelineFilesEnum.best_near_fit_vectorial_lightcurve,
            PipelineFilesEnum.best_far_fit_vectorial_lightcurve,
            PipelineFilesEnum.best_full_fit_vectorial_lightcurve,
        ]
        pipeline_file_exists = functools.partial(
            self.pipeline_files.exists,
            epoch_id=epoch_id,
            stacking_method=stacking_method,
        )
        products_exist = list(map(pipeline_file_exists, products))

        if all(products_exist):
            return SwiftCometPipelineStepStatus.complete
        if all([not x for x in products_exist]):
            return SwiftCometPipelineStepStatus.not_complete
        return SwiftCometPipelineStepStatus.partial

    def can_step_be_run(
        self,
        step: SwiftCometPipelineStepEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
    ) -> bool:
        require = self.steps[step].require
        if require is None:
            return True

        prev_step_status = self.get_status(
            require,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )

        if prev_step_status == SwiftCometPipelineStepStatus.complete:
            return True

        return False

    # TODO: function to recurse through the requirements to check for inconsistent pipeline state

    def get_product(
        self,
        pf: PipelineFilesEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        fit_type: VectorialFitType | None = None,
    ) -> PipelineProduct | None:
        return self.pipeline_files.get_product(
            pf=pf,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            # fit_type=fit_type,
        )

    def get_product_data(
        self,
        pf: PipelineFilesEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        # fit_type: VectorialFitType | None = None,
    ) -> Any | None:
        p = self.pipeline_files.get_product(
            pf=pf,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            # fit_type=fit_type,
        )
        if p is None:
            return None
        p.read_product_if_not_loaded()
        return p.data

    def get_epoch_id_list(self) -> list[EpochID] | None:
        return self.pipeline_files.get_epoch_id_list()

    def exists(
        self,
        pf: PipelineFilesEnum,
        epoch_id: EpochID | None = None,
        filter_type: SwiftFilter | None = None,
        stacking_method: StackingMethod | None = None,
        # fit_type: VectorialFitType | None = None,
    ) -> bool:

        return self.pipeline_files.exists(
            pf=pf,
            epoch_id=epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
            # fit_type=fit_type,
        )

    def has_epoch_been_stacked(self, epoch_id: EpochID) -> bool:
        filters = [SwiftFilter.uw1, SwiftFilter.uvv]
        sms = [StackingMethod.summation, StackingMethod.median]

        return all(
            [
                self.exists(
                    pf=PipelineFilesEnum.stacked_image,
                    epoch_id=epoch_id,
                    filter_type=f,
                    stacking_method=s,
                )
                for f, s in product(filters, sms)
            ]
        )
