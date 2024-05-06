import itertools
import pathlib
import calendar
import yaml
import glob
from abc import ABC, abstractmethod
from typing import Any, Optional, List
from itertools import product

from icecream import ic
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.time import Time

from swift_comet_pipeline.observationlog.epochs import (
    epoch_stacked_image_to_fits,
    read_epoch,
    write_epoch,
    Epoch,
)
from swift_comet_pipeline.observationlog.observation_log import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.swift.swift_data import SwiftData
from swift_comet_pipeline.swift.swift_filter import (
    SwiftFilter,
    filter_to_file_string,
    filter_to_string,
)
from swift_comet_pipeline.stacking.stacking import (
    StackedUVOTImageSet,
    StackingMethod,
    stack_epoch_into_sum_and_median,
)
from swift_comet_pipeline.swift.uvot_image import pad_to_match_sizes


# TODO: find out why stacked epoch products are saying every entry is uw1 filter


class PipelineProduct(ABC):
    """
    Base class for files produced by the pipeline - given a path product_path, this object is responsible
    for loading and saving data
    """

    def __init__(self, product_path: pathlib.Path):
        self.product_path = product_path
        self._data: Optional[Any] = None

    @property
    def data(self) -> Optional[Any]:
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    def exists(self) -> bool:
        return self.product_path.exists()

    @abstractmethod
    def read(self) -> None:
        if not self.exists():
            ic(f"Request to read product {self.product_path} but file does not exist!")
            return

    @abstractmethod
    def write(self) -> None:
        if self._data is None:
            ic(
                f"Request to write product {self.product_path} with no data to write! Skipping."
            )

    def delete_file(self) -> None:
        if self.exists():
            self.product_path.unlink()


class YAMLDictPipelineProductIO(PipelineProduct):
    """
    Product for dict <----> yaml file
    """

    def read(self) -> None:
        super().read()
        with open(self.product_path, "r") as stream:
            try:
                read_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                read_yaml = None
                ic(f"Reading file {self.product_path} resulted in yaml error {e}")

        self._data = read_yaml

    def write(self) -> None:
        super().write()

        if self._data is None:
            return

        with open(self.product_path, "w") as stream:
            try:
                yaml.safe_dump(self._data, stream)
            except yaml.YAMLError as e:
                ic(e)


class CSVDataframePipelineProductIO(PipelineProduct):
    """
    Product for pd.DataFrame <----> csv file
    """

    def read(self) -> None:
        super().read()
        self._data = pd.read_csv(self.product_path)

    def write(self) -> None:
        super().write()
        if self._data is not None:
            self._data.to_csv(self.product_path, index=False)


# begin data ingestion pipeline


# begin observation log step
class ObservationLogProductIO(PipelineProduct):
    """
    For saving/loading an observation log, which needs its own methods to process data types before writing and after reading
    """

    def read(self) -> None:
        super().read()
        self._data = read_observation_log(obs_log_path=self.product_path)

    def write(self) -> None:
        super().write()
        if self._data is not None:
            write_observation_log(obs_log=self._data, obs_log_path=self.product_path)


class ObservationLogProduct(ObservationLogProductIO):
    """
    For saving/loading an observation log, which needs its own methods to process data types before writing and after reading
    """

    def __init__(self, product_path: pathlib.Path):
        super().__init__(product_path=product_path)

        self.product_path = self.product_path / pathlib.Path("observation_log.parquet")


# end observation log step


# begin orbital data step
class OrbitalDataProduct(CSVDataframePipelineProductIO):
    """
    For saving/loading orbital data in csv format - children should append filename to self.product_path
    """

    def __init__(self, product_path: pathlib.Path):
        super().__init__(product_path=product_path)

        self.product_path = self.product_path / "orbital_data"
        self.product_path.mkdir(parents=True, exist_ok=True)


class EarthOrbitalDataProduct(OrbitalDataProduct):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.product_path = self.product_path / pathlib.Path(
            "horizons_earth_orbital_data.csv"
        )


class CometOrbitalDataProduct(OrbitalDataProduct):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.product_path = self.product_path / pathlib.Path(
            "horizons_comet_orbital_data.csv"
        )


# end orbital data step


# begin epoch slicing step
class EpochProductIO(PipelineProduct):
    """
    For saving/loading an epoch, which needs its own methods to process data types before writing and after reading
    """

    def read(self) -> None:
        super().read()
        self._data = read_epoch(epoch_path=self.product_path)

    def write(self) -> None:
        super().write()
        if self._data is not None:
            write_epoch(epoch=self._data, epoch_path=self.product_path)


class EpochProduct(EpochProductIO):
    """
    We name the epochs as a batch with their time-ordered index prepended, like 000_Jan_01_2020 - so these products cannot name themselves
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.epoch_id = self.product_path.stem


# end epoch slicing step

# End data ingestion pipeline


class FitsImageProductIO(PipelineProduct):
    """
    instead of fighting fits file formatting and HDU lists, we assert that hdu[0] is an empty primary HDU
    and that hdu[1] is an ImageHDU, with relevant header and image data, for the images we create via
    the pipeline

    This is NOT a general-purpose FITS reader/writer!
    """

    def write(self) -> None:
        super().write()
        if self._data is not None:
            self._data.writeto(self.product_path, overwrite=True)

    def read(self) -> None:
        hdul = fits.open(self.product_path, lazy_load_hdus=False, memmap=True)
        self._data = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()


# Begin epoch processing pipeline


class EpochPipelineProduct(PipelineProduct):
    """
    Base class for products in the sub-pipeline that we run for each epoch that we pull from the observation log
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parent_epoch = parent_epoch
        self.filter_type = filter_type
        self.stacking_method = stacking_method


# image stacking step
class EpochPipelineStackingProduct(EpochPipelineProduct):
    """
    Base class for products in the stacking pipeline step - children only need to call super().__init__() and then modify the product_path to add their filename
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: Optional[SwiftFilter],
        stacking_method: Optional[StackingMethod],
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )
        self.product_path = (
            self.product_path / "stacked" / parent_epoch.product_path.stem
        )
        self.product_path.mkdir(parents=True, exist_ok=True)


class StackedEpochProduct(EpochPipelineStackingProduct, EpochProduct):
    """
    These epochs have all DataFrame rows removed that were not included in image stacking, like images taken in non-UW1 or non-UVV filters.
    This removes the need for logic to check if any particular dataframe row was included or not - we can read this epoch and assume every image mentioned is included.
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=None,
            stacking_method=None,
            *args,
            **kwargs,
        )

        # use the same name as the parent epoch
        self.product_path = self.product_path / parent_epoch.product_path.name


class StackedFitsImageProduct(EpochPipelineStackingProduct, FitsImageProductIO):

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        fits_filename = (
            f"{parent_epoch.product_path.stem}_{filter_string}_{stacking_method}.fits"
        )

        self.product_path = self.product_path / pathlib.Path(fits_filename)


# end stacking step


# begin analysis steps
class EpochPipelineAnalysisProduct(EpochPipelineProduct):
    """
    base class for epoch sub-pipeline analysis products - children should call super().__init()__ and the modify self.product_path to append their file name
    """

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )
        self.product_path = (
            self.product_path / "analysis" / parent_epoch.product_path.stem
        )
        self.product_path.mkdir(parents=True, exist_ok=True)


# begin background subtraction step
class BackgroundAnalysisProduct(
    EpochPipelineAnalysisProduct, YAMLDictPipelineProductIO
):

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        bga_filename = f"background_analysis_{filter_string}_{stacking_method}.yaml"

        self.product_path = self.product_path / pathlib.Path(bga_filename)


class BackgroundSubtractedFITSProduct(EpochPipelineAnalysisProduct, FitsImageProductIO):

    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        fits_filename = f"bg_subtracted_{filter_string}_{stacking_method}.fits"

        self.product_path = self.product_path / pathlib.Path(fits_filename)


# end background subtraction step


# begin Q(H2O) vs aperture radius step
class QvsApertureRadiusProduct(
    EpochPipelineAnalysisProduct, CSVDataframePipelineProductIO
):

    def __init__(
        self,
        parent_epoch: EpochProduct,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        qvsa_filename = f"qh2o_vs_aperture_radius_{stacking_method}.csv"
        self.product_path = self.product_path / pathlib.Path(qvsa_filename)


# end Q(H2O) vs aperture radius step


# begin Q from profile extraction step
class QFromProfileExtractionAnalysis(
    EpochPipelineAnalysisProduct, YAMLDictPipelineProductIO
):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        qfpe_filename = f"qh2o_from_profile_extraction_{stacking_method}.yaml"
        self.product_path = self.product_path / pathlib.Path(qfpe_filename)


class ExtractedRadialProfile(
    EpochPipelineAnalysisProduct, CSVDataframePipelineProductIO
):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        ep_filename = f"extracted_profile_{filter_string}_{stacking_method}.csv"
        self.product_path = self.product_path / pathlib.Path(ep_filename)


class ExtractedRadialProfileImage(EpochPipelineAnalysisProduct, FitsImageProductIO):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        epi_filename = f"extracted_profile_image_{filter_string}_{stacking_method}.fits"
        self.product_path = self.product_path / pathlib.Path(epi_filename)


class MedianSubtractedImage(EpochPipelineAnalysisProduct, FitsImageProductIO):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        msi_filename = f"median_subtracted_image_{filter_string}_{stacking_method}.fits"
        self.product_path = self.product_path / pathlib.Path(msi_filename)


class MedianDividedImage(EpochPipelineAnalysisProduct, FitsImageProductIO):
    def __init__(
        self,
        parent_epoch: EpochProduct,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent_epoch=parent_epoch,
            filter_type=filter_type,
            stacking_method=stacking_method,
            *args,
            **kwargs,
        )

        filter_string = filter_to_file_string(filter_type=filter_type)
        mdi_filename = f"median_divided_image_{filter_string}_{stacking_method}.fits"
        self.product_path = self.product_path / pathlib.Path(mdi_filename)


# end Q from profile extraction step


# class PipelineProductType(StrEnum):
#     observation_log = auto()
#     comet_orbital_data = auto()
#     earth_orbital_data = auto()
#     epoch = auto()
#     stacked_epoch = auto()
#     stacked_image = auto()
#     stacked_image_header = auto()
#     background_analysis = auto()
#     background_subtracted_image = auto()
#     qh2o_vs_aperture_radius = auto()
#     qh2o_from_profile = auto()


# class PipelineStep(Protocol):
#     def dependencies(self) -> List[PipelineProductType]: ...
#     def products(self) -> List[PipelineProductType]: ...


# class DataIngestionPipelineStep(StrEnum):
#     observation_log = auto()
#     comet_orbital_data = auto()
#     earth_orbital_data = auto()
#     epoch_slicing = auto()
#     data_ingestion_complete = auto()
#
#
# class EpochProcessingPipelineStep(StrEnum):
#     image_stacking = auto()
#     background_subtraction = auto()
#     qvsr_analysis = auto()
#     q_from_profile = auto()


class DataIngestionFiles:
    """
    This data ingestion step gathers all of the data into a database, splits the
    observation log into epochs, and prepares the epochs for analysis
    """

    def __init__(self, project_path: pathlib.Path):

        self.project_path: pathlib.Path = project_path
        self.epoch_dir_path: pathlib.Path = self.project_path / pathlib.Path("epochs")

        self.observation_log = ObservationLogProduct(product_path=self.project_path)
        self.comet_orbital_data = CometOrbitalDataProduct(
            product_path=self.project_path
        )
        self.earth_orbital_data = EarthOrbitalDataProduct(
            product_path=self.project_path
        )

        self.epochs: Optional[List[EpochProduct]] = None
        self.scan_for_epoch_files()

    def _find_epoch_files(self) -> Optional[List[pathlib.Path]]:
        """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
        glob_pattern = str(self.epoch_dir_path / pathlib.Path("*.parquet"))
        epoch_filename_list = sorted(glob.glob(glob_pattern))
        if len(epoch_filename_list) == 0:
            return None
        return [pathlib.Path(x) for x in epoch_filename_list]

    def scan_for_epoch_files(self):
        """
        Look on disk for existing epochs and create EpochProducts out of them if they are there
        """
        epoch_path_list = self._find_epoch_files()
        if epoch_path_list is not None:
            self.epochs = [EpochProduct(product_path=x) for x in epoch_path_list]

    def create_epochs(self, epoch_list: List[Epoch], write_to_disk: bool) -> None:
        """
        The epochs should be time-sorted before they are passed in!
        """
        epoch_path_list = []
        for i, epoch in enumerate(epoch_list):
            epoch_mid = Time(np.min(epoch.MID_TIME)).ymdhms
            day = epoch_mid.day  # type: ignore
            month = calendar.month_abbr[epoch_mid.month]  # type: ignore
            year = epoch_mid.year  # type: ignore

            epoch_path_list.append(
                pathlib.Path(f"{i:03d}_{year}_{day:02d}_{month}.parquet")
            )

        self.epochs = [
            EpochProduct(product_path=self.epoch_dir_path / x) for x in epoch_path_list
        ]

        for epoch_product, epoch in zip(self.epochs, epoch_list):
            epoch_product.data = epoch
            if write_to_disk:
                epoch_product.write()

    def delete_epochs(self) -> None:
        if not self.epochs:
            return

        for epoch_product in self.epochs:
            epoch_product.delete_file()

        self.epochs = None

    # TODO:
    # def next_step(self):
    #     for d in dependencies:
    #         if not d.satisfied:
    #             return d
    #
    #     return DataIngestionPipelineStep.data_ingestion_complete


class EpochSubpipelineFiles:
    """
    We have a pipeline to run on every individual epoch that we have prepared with the data ingestion step
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
                uw1=stacked_images[(SwiftFilter.uw1, stacking_method)],
                uvv=stacked_images[(SwiftFilter.uvv, stacking_method)],
            )
            stacked_images[(SwiftFilter.uw1, stacking_method)] = uw1_img
            stacked_images[(SwiftFilter.uvv, stacking_method)] = uvv_img

        # push all the data into the products for writing later
        self.stacked_epoch.data = epoch_to_stack
        for filter_type, stacking_method in itertools.product(
            uw1_and_uvv, sum_and_median
        ):
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

        for filter_type, stacking_method in itertools.product(
            uw1_and_uvv, sum_and_median
        ):
            self.stacked_images[filter_type, stacking_method].write()

    def get_stacked_image_set(self) -> Optional[StackedUVOTImageSet]:
        stacked_image_set = {}

        uw1_and_uvv = [SwiftFilter.uvv, SwiftFilter.uw1]
        sum_and_median = [StackingMethod.summation, StackingMethod.median]

        # TODO: we should check if any of the data is None and return None if so - we don't have a valid set of stacked images somehow
        for filter_type, stacking_method in itertools.product(
            uw1_and_uvv, sum_and_median
        ):
            if self.stacked_images[filter_type, stacking_method].data is None:
                self.stacked_images[filter_type, stacking_method].read()

            # the 'data' of the product includes a data.header for the FITS header, and data.data for the numpy image array
            stacked_image_set[filter_type, stacking_method] = self.stacked_images[
                filter_type, stacking_method
            ].data.data

        return stacked_image_set


class PipelineFiles:
    def __init__(self, project_path: pathlib.Path):
        self.project_path = project_path
        self.data_ingestion_files = DataIngestionFiles(project_path=self.project_path)

        self.epoch_subpipelines: Optional[List[EpochSubpipelineFiles]] = None
        if self.data_ingestion_files.epochs is None:
            return

        # TODO: just make this a dict of parent EpochProduct -> EpochSubpipelineFiles
        self.epoch_subpipelines = [
            EpochSubpipelineFiles(project_path=self.project_path, parent_epoch=x)
            for x in self.data_ingestion_files.epochs
        ]

    def epoch_subpipeline_from_parent_epoch(
        self, parent_epoch: EpochProduct
    ) -> Optional[EpochSubpipelineFiles]:
        if self.epoch_subpipelines is None:
            return None

        matching_subs = [
            x for x in self.epoch_subpipelines if x.parent_epoch == parent_epoch
        ]
        assert len(matching_subs) == 1

        return matching_subs[0]
