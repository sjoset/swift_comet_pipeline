import calendar
import pathlib
import glob
import yaml
from enum import StrEnum, auto
from typing import List, Optional, Any, Tuple, TypeAlias
from itertools import product

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.io import fits

from swift_comet_pipeline.pipeline.determine_background import (
    BackgroundResult,
    dict_to_background_result,
)
from swift_comet_pipeline.observationlog.epochs import Epoch, read_epoch, write_epoch
from swift_comet_pipeline.observationlog.observation_log import (
    SwiftObservationLog,
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.swift.swift_filter import filter_to_file_string, SwiftFilter
from swift_comet_pipeline.swift.uvot_image import SwiftUVOTImage
from swift_comet_pipeline.stacking.stacking import StackingMethod


__all__ = ["PipelineFiles"]


class PipelineProductType(StrEnum):
    observation_log = auto()
    comet_orbital_data = auto()
    earth_orbital_data = auto()
    epoch = auto()
    stacked_epoch = auto()
    stacked_image = auto()
    stacked_image_header = auto()
    background_analysis = auto()
    background_subtracted_image = auto()
    qh2o_vs_aperture_radius = auto()
    qh2o_from_profile = auto()


# The epoch ID will be the leading part of the filename: XXX_Year_day_month
PipelineEpochID: TypeAlias = str


class PipelineFiles:
    def __init__(self, base_product_save_path: pathlib.Path):
        """Construct file paths to pipeline products for later loading/saving"""
        self.base_product_save_path: pathlib.Path = base_product_save_path
        self.epoch_dir_path: pathlib.Path = self._construct_epoch_dir_path()
        self.stack_dir_path: pathlib.Path = self._construct_stack_dir_path()
        self.analysis_base_path: pathlib.Path = self._construct_analysis_base_path()
        self.orbital_data_path: pathlib.Path = self._construct_orbital_data_path()
        self._create_directories()

        self.observation_log_path: pathlib.Path = self._construct_observation_log_path()

        self.comet_orbital_data_path: pathlib.Path = (
            self._construct_comet_orbital_data_path()
        )
        self.earth_orbital_data_path: pathlib.Path = (
            self._construct_earth_orbital_data_path()
        )

        self.epoch_ids: Optional[List[PipelineEpochID]] = None
        self.epoch_paths: Optional[dict[PipelineEpochID, pathlib.Path]] = None
        self.stacked_epoch_path: Optional[dict[PipelineEpochID, pathlib.Path]] = None
        self.stacked_image_path: Optional[
            dict[Tuple[PipelineEpochID, SwiftFilter, StackingMethod], pathlib.Path]
        ] = None
        self.background_analysis_path: Optional[
            dict[Tuple[PipelineEpochID, SwiftFilter, StackingMethod], pathlib.Path]
        ] = None
        self.background_subtracted_image_path: Optional[
            dict[Tuple[PipelineEpochID, SwiftFilter, StackingMethod], pathlib.Path]
        ] = None
        self.qh2o_vs_aperture_radius_path: Optional[
            dict[Tuple[PipelineEpochID, StackingMethod], pathlib.Path]
        ] = None
        self.qh2o_from_profile_path: Optional[
            dict[Tuple[PipelineEpochID, StackingMethod], pathlib.Path]
        ] = None

        epoch_path_list = self._find_epoch_files()
        if epoch_path_list is None:
            return
        self._initialize_epochs_from_path_list(epoch_path_list)
        self._initialize_products(epoch_ids=self.epoch_ids)  # type: ignore

    def _construct_observation_log_path(self) -> pathlib.Path:
        return self.base_product_save_path / pathlib.Path("observation_log.parquet")

    def _construct_orbital_data_path(self) -> pathlib.Path:
        return self.base_product_save_path / pathlib.Path("orbital_data")

    def _construct_comet_orbital_data_path(self) -> pathlib.Path:
        return self._construct_orbital_data_path() / pathlib.Path(
            "horizons_comet_orbital_data.csv"
        )

    def _construct_earth_orbital_data_path(self) -> pathlib.Path:
        return self._construct_orbital_data_path() / pathlib.Path(
            "horizons_earth_orbital_data.csv"
        )

    def _construct_epoch_dir_path(self) -> pathlib.Path:
        return self.base_product_save_path / pathlib.Path("epochs")

    def _construct_stack_dir_path(self) -> pathlib.Path:
        return self.base_product_save_path / pathlib.Path("stacked")

    def _construct_analysis_base_path(self) -> pathlib.Path:
        return self.base_product_save_path / pathlib.Path("analysis")

    def _create_directories(self) -> None:
        for p in [
            self.orbital_data_path,
            self.epoch_dir_path,
            self.stack_dir_path,
            self.analysis_base_path,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def _find_epoch_files(self) -> Optional[List[pathlib.Path]]:
        """If there are epoch files generated for this project, return a list of paths to them, otherwise None"""
        glob_pattern = str(self.epoch_dir_path / pathlib.Path("*.parquet"))
        epoch_filename_list = sorted(glob.glob(glob_pattern))
        if len(epoch_filename_list) == 0:
            return None
        return [pathlib.Path(x) for x in epoch_filename_list]

    def _get_epoch_id(self, epoch_path: pathlib.Path) -> PipelineEpochID:
        """The stem is just the file name, without leading path or file extension"""
        return epoch_path.stem

    def _epoch_id_to_epoch_path(self, epoch_id: PipelineEpochID) -> pathlib.Path:
        return self.epoch_dir_path / pathlib.Path(epoch_id + ".parquet")

    def _initialize_epochs_from_path_list(
        self, epoch_path_list: List[pathlib.Path]
    ) -> None:
        self.epoch_ids = [self._get_epoch_id(x) for x in epoch_path_list]
        self.epoch_paths = dict(zip(self.epoch_ids, epoch_path_list))

    def _initialize_products(self, epoch_ids: List[PipelineEpochID]) -> None:
        stacked_epoch_paths = [self._construct_stacked_epoch_path(x) for x in epoch_ids]
        self.stacked_epoch_path = dict(zip(epoch_ids, stacked_epoch_paths))

        stacked_image_dict = {}
        bg_analysis_dict = {}
        bg_sub_dict = {}
        for epoch_id, filter_type, stacking_method in product(
            epoch_ids,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            stacked_image_dict[epoch_id, filter_type, stacking_method] = (
                self._construct_stacked_image_path(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )
            bg_analysis_dict[epoch_id, filter_type, stacking_method] = (
                self._construct_background_analysis_path(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )
            bg_sub_dict[epoch_id, filter_type, stacking_method] = (
                self._construct_background_subtracted_image_path(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            )

        self.stacked_image_path = stacked_image_dict
        self.background_analysis_path = bg_analysis_dict
        self.background_subtracted_image_path = bg_sub_dict

        qh2o_vs_r_dict = {}
        qh2o_from_profile_dict = {}
        for epoch_id, stacking_method in product(
            epoch_ids, [StackingMethod.summation, StackingMethod.median]
        ):
            qh2o_vs_r_dict[epoch_id, stacking_method] = (
                self._construct_qh2o_vs_aperture_path(
                    source_epoch_id=epoch_id, stacking_method=stacking_method
                )
            )
            qh2o_from_profile_dict[epoch_id, stacking_method] = (
                self._construct_qh2o_from_profile_path(
                    source_epoch_id=epoch_id, stacking_method=stacking_method
                )
            )

        self.qh2o_vs_aperture_radius_path = qh2o_vs_r_dict
        self.qh2o_from_profile_path = qh2o_from_profile_dict

    def _construct_stacked_epoch_path(
        self, source_epoch_id: PipelineEpochID
    ) -> pathlib.Path:
        """Epoch saved after having entries removed (marked as veto, etc.) are stored with the same name, but in the stack folder"""
        # source_epoch_path.name is the file name without the leading path, but includes file extension (.parquet)
        return self.stack_dir_path / pathlib.Path(source_epoch_id + ".parquet")

    def _construct_stacked_image_path(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> pathlib.Path:
        """Image produced from stacking the source epoch, by filter and stack type"""
        filter_string = filter_to_file_string(filter_type=filter_type)
        fits_filename = f"{source_epoch_id}_{filter_string}_{stacking_method}.fits"

        return self.stack_dir_path / pathlib.Path(fits_filename)

    def _epoch_analysis_path(self, source_epoch_id: PipelineEpochID) -> pathlib.Path:
        return self.analysis_base_path / pathlib.Path(source_epoch_id)

    def _construct_background_analysis_path(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> pathlib.Path:
        filter_string = filter_to_file_string(filter_type=filter_type)
        bg_filename = f"background_analysis_{filter_string}_{stacking_method}.yaml"
        return self._epoch_analysis_path(
            source_epoch_id=source_epoch_id
        ) / pathlib.Path(bg_filename)

    def _construct_background_subtracted_image_path(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> pathlib.Path:
        filter_string = filter_to_file_string(filter_type=filter_type)
        fits_filename = f"bg_subtracted_{filter_string}_{stacking_method}.fits"
        return self._epoch_analysis_path(
            source_epoch_id=source_epoch_id
        ) / pathlib.Path(fits_filename)

    def _construct_qh2o_vs_aperture_path(
        self, source_epoch_id: PipelineEpochID, stacking_method: StackingMethod
    ) -> pathlib.Path:
        return self._epoch_analysis_path(
            source_epoch_id=source_epoch_id
        ) / pathlib.Path(f"qh2o_vs_aperture_radius_{stacking_method}.csv")

    def _construct_qh2o_from_profile_path(
        self, source_epoch_id: PipelineEpochID, stacking_method: StackingMethod
    ) -> pathlib.Path:
        return self._epoch_analysis_path(
            source_epoch_id=source_epoch_id
        ) / pathlib.Path(f"qh2o_from_profile_{stacking_method}.csv")

    def get_epoch_ids(self) -> Optional[List[PipelineEpochID]]:
        if self.epoch_ids is None:
            return None
        return self.epoch_ids

    def create_epochs(self, epoch_list: List[Epoch]) -> None:
        epoch_path_list = []
        for i, epoch in enumerate(epoch_list):
            epoch_mid = Time(np.min(epoch.MID_TIME)).ymdhms
            day = epoch_mid.day
            month = calendar.month_abbr[epoch_mid.month]
            year = epoch_mid.year

            filename = pathlib.Path(f"{i:03d}_{year}_{day:02d}_{month}.parquet")
            epoch_path_list.append(self.epoch_dir_path / filename)

        if self.epoch_paths is not None or self.epoch_ids is not None:
            print(
                "warning: attempting to create epochs after detecting existing epochs"
            )

        self._initialize_epochs_from_path_list(epoch_path_list=epoch_path_list)

        for epoch, epoch_id in zip(epoch_list, self.epoch_ids):  # type: ignore
            self.write_pipeline_product(
                PipelineProductType.epoch, data=epoch, epoch_id=epoch_id
            )

    def read_pipeline_product(
        self,
        p: PipelineProductType,
        epoch_id: Optional[PipelineEpochID] = None,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
    ) -> Optional[Any]:
        match p:  # type: ignore
            case PipelineProductType.observation_log:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                return self._read_observation_log()
            case PipelineProductType.comet_orbital_data:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                return self._read_comet_orbital_data()
            case PipelineProductType.earth_orbital_data:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                return self._read_earth_orbital_data()
            case PipelineProductType.epoch:
                if filter_type is not None or stacking_method is not None:
                    print(
                        "Warning: received arguments {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                if epoch_id is None:
                    return None
                return self._read_epoch(epoch_id=epoch_id)
            case PipelineProductType.stacked_epoch:
                if filter_type is not None or stacking_method is not None:
                    print(
                        "Warning: received arguments {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                if epoch_id is None:
                    return None
                return self._read_stacked_epoch(source_epoch_id=epoch_id)
            case PipelineProductType.stacked_image:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                return self._read_stacked_image(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case PipelineProductType.stacked_image_header:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                return self._read_stacked_image_header(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case PipelineProductType.background_analysis:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                return self._read_background_analysis(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case PipelineProductType.background_subtracted_image:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                return self._read_background_subtracted_image(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                )
            case PipelineProductType.qh2o_vs_aperture_radius:
                if epoch_id is None or stacking_method is None:
                    return None
                if filter_type is not None:
                    print(
                        "Warning: received arguments {filter_type=} when unnecessary!"
                    )
                return self._read_qh2o_vs_aperture(
                    source_epoch_id=epoch_id, stacking_method=stacking_method
                )
            case PipelineProductType.qh2o_from_profile:
                if epoch_id is None or stacking_method is None:
                    return None
                if filter_type is not None:
                    print(
                        "Warning: received arguments {filter_type=} when unnecessary!"
                    )
                return self._read_qh2o_from_profile(
                    source_epoch_id=epoch_id, stacking_method=stacking_method
                )

    def write_pipeline_product(
        self,
        p: PipelineProductType,
        data: Any,
        epoch_id: Optional[PipelineEpochID] = None,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
    ) -> None:
        match p:  # type: ignore
            case PipelineProductType.observation_log:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                self._write_observation_log(obs_log=data)
            case PipelineProductType.comet_orbital_data:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                self._write_comet_orbital_data(df=data)
            case PipelineProductType.earth_orbital_data:
                if (
                    filter_type is not None
                    or stacking_method is not None
                    or epoch_id is not None
                ):
                    print(
                        "Warning: received arguments {epoch_path=}, {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                self._write_earth_orbital_data(df=data)
            case PipelineProductType.epoch:
                if filter_type is not None or stacking_method is not None:
                    print(
                        "Warning: received arguments {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                if epoch_id is None:
                    return None
                self._write_epoch(epoch_id=epoch_id, epoch=data)
            case PipelineProductType.stacked_epoch:
                if filter_type is not None or stacking_method is not None:
                    print(
                        "Warning: received arguments {filter_type=}, {stacking_method=} when unnecessary!"
                    )
                if epoch_id is None:
                    return None
                self._write_stacked_epoch(source_epoch_id=epoch_id, stacked_epoch=data)
            case PipelineProductType.stacked_image:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                self._write_stacked_image(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                    img_hdu=data,
                )
            case PipelineProductType.stacked_image_header:
                # not supported to write just the header and keep the image data
                return
            case PipelineProductType.background_analysis:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                self._write_background_analysis(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                    yaml_dict=data,
                )
            case PipelineProductType.background_subtracted_image:
                if epoch_id is None or filter_type is None or stacking_method is None:
                    return None
                self._write_background_subtracted_image(
                    source_epoch_id=epoch_id,
                    filter_type=filter_type,
                    stacking_method=stacking_method,
                    img_hdu=data,
                )
            case PipelineProductType.qh2o_vs_aperture_radius:
                if epoch_id is None or stacking_method is None:
                    return None
                if filter_type is not None:
                    print(
                        "Warning: received arguments {filter_type=} when unnecessary!"
                    )
                self._write_qh2o_vs_aperture(
                    source_epoch_id=epoch_id, stacking_method=stacking_method, df=data
                )
            # TODO
            # case PipelineProductType.qh2o_from_profile:
            #     if epoch_id is None or stacking_method is None:
            #         return None
            #     if filter_type is not None:
            #         print(
            #             "Warning: received arguments {filter_type=} when unnecessary!"
            #         )
            #     return self._read_qh2o_from_profile(source_epoch_id=epoch_id, stacking_method=stacking_method)

    def _read_observation_log(self) -> SwiftObservationLog:
        return read_observation_log(self.observation_log_path)

    def _write_observation_log(self, obs_log: SwiftObservationLog) -> None:
        write_observation_log(obs_log=obs_log, obs_log_path=self.observation_log_path)

    def _read_comet_orbital_data(self) -> pd.DataFrame:
        return pd.read_csv(self.comet_orbital_data_path)

    def _write_comet_orbital_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.comet_orbital_data_path)

    def _read_earth_orbital_data(self) -> pd.DataFrame:
        return pd.read_csv(self.earth_orbital_data_path)

    def _write_earth_orbital_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.earth_orbital_data_path)

    def _read_epoch(self, epoch_id: PipelineEpochID) -> Epoch:
        return read_epoch(epoch_path=self._epoch_id_to_epoch_path(epoch_id=epoch_id))

    def _write_epoch(self, epoch_id: PipelineEpochID, epoch: Epoch) -> None:
        if self.epoch_paths is None:
            return
        write_epoch(epoch=epoch, epoch_path=self.epoch_paths[epoch_id])

    def _read_stacked_epoch(self, source_epoch_id: PipelineEpochID) -> Epoch:
        return read_epoch(
            epoch_path=self._construct_stacked_epoch_path(
                source_epoch_id=source_epoch_id
            )
        )

    def _write_stacked_epoch(
        self, source_epoch_id: PipelineEpochID, stacked_epoch: Epoch
    ) -> None:
        if self.epoch_paths is None or self.stacked_epoch_path is None:
            return
        write_epoch(
            epoch=stacked_epoch, epoch_path=self.stacked_epoch_path[source_epoch_id]
        )

    def _read_stacked_image(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> SwiftUVOTImage:
        img_path = self._construct_stacked_image_path(
            source_epoch_id=source_epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        hdul = fits.open(img_path, lazy_load_hdus=False, memmap=True)
        img_data = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img_data.data  # type: ignore

    def _write_stacked_image(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        img_hdu,
    ) -> None:
        if self.stacked_image_path is None:
            return
        path = self.stacked_image_path[source_epoch_id, filter_type, stacking_method]
        img_hdu.writeto(path, overwrite=True)

    def _read_stacked_image_header(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ):
        img_path = self._construct_stacked_image_path(
            source_epoch_id=source_epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        hdul = fits.open(img_path, lazy_load_hdus=False, memmap=True)
        img_data = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img_data.header

    def _read_background_analysis(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> Optional[BackgroundResult]:
        bg_path = self._construct_background_analysis_path(
            source_epoch_id=source_epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        print(bg_path)
        with open(bg_path, "r") as stream:
            try:
                raw_yaml = yaml.safe_load(stream)
                return dict_to_background_result(raw_yaml=raw_yaml)
            except yaml.YAMLError as exc:
                print(exc)
                return None

    def _write_background_analysis(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        yaml_dict: dict,
    ) -> None:
        if self.background_analysis_path is None:
            return
        bg_path = self.background_analysis_path[
            source_epoch_id, filter_type, stacking_method
        ]
        bg_path.parent.mkdir(exist_ok=True, parents=True)
        with open(bg_path, "w") as stream:
            try:
                yaml.safe_dump(yaml_dict, stream)
            except yaml.YAMLError as exc:
                print(exc)

    def _read_background_subtracted_image(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
    ) -> SwiftUVOTImage:
        img_path = self._construct_background_subtracted_image_path(
            source_epoch_id=source_epoch_id,
            filter_type=filter_type,
            stacking_method=stacking_method,
        )
        hdul = fits.open(img_path, lazy_load_hdus=False, memmap=True)
        img_data = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
        return img_data.data  # type: ignore

    def _write_background_subtracted_image(
        self,
        source_epoch_id: PipelineEpochID,
        filter_type: SwiftFilter,
        stacking_method: StackingMethod,
        img_hdu,
    ) -> None:
        if self.background_subtracted_image_path is None:
            return
        img_path = self.background_subtracted_image_path[
            source_epoch_id, filter_type, stacking_method
        ]
        img_hdu.writeto(img_path, overwrite=True)

    def _read_qh2o_vs_aperture(
        self, source_epoch_id: PipelineEpochID, stacking_method: StackingMethod
    ) -> pd.DataFrame:
        data_path = self._construct_qh2o_vs_aperture_path(
            source_epoch_id=source_epoch_id, stacking_method=stacking_method
        )
        return pd.read_csv(data_path)

    def _write_qh2o_vs_aperture(
        self,
        source_epoch_id: PipelineEpochID,
        stacking_method: StackingMethod,
        df: pd.DataFrame,
    ) -> None:
        if self.qh2o_vs_aperture_radius_path is None:
            return
        df_path = self.qh2o_vs_aperture_radius_path[source_epoch_id, stacking_method]
        df.to_csv(df_path)

    def _read_qh2o_from_profile(
        self, source_epoch_id: PipelineEpochID, stacking_method: StackingMethod
    ) -> pd.DataFrame:
        data_path = self._construct_qh2o_from_profile_path(
            source_epoch_id=source_epoch_id, stacking_method=stacking_method
        )
        return pd.read_csv(data_path)

    def exists(
        self,
        p: PipelineProductType,
        epoch_id: Optional[PipelineEpochID] = None,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
    ) -> bool:
        match p:  # type: ignore
            case PipelineProductType.observation_log:
                path = self.observation_log_path
            case PipelineProductType.comet_orbital_data:
                path = self.comet_orbital_data_path
            case PipelineProductType.earth_orbital_data:
                path = self.earth_orbital_data_path
            case PipelineProductType.epoch:
                if self.epoch_paths is None or epoch_id is None:
                    return False
                path = self.epoch_paths[epoch_id]
            case PipelineProductType.stacked_epoch:
                if self.stacked_epoch_path is None or epoch_id is None:
                    return False
                path = self.stacked_epoch_path[epoch_id]
            case PipelineProductType.stacked_image:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.stacked_image_path is None
                ):
                    return False
                path = self.stacked_image_path[epoch_id, filter_type, stacking_method]
            case PipelineProductType.stacked_image_header:
                return False
            case PipelineProductType.background_analysis:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.background_analysis_path is None
                ):
                    return False
                path = self.background_analysis_path[
                    epoch_id, filter_type, stacking_method
                ]
            case PipelineProductType.background_subtracted_image:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.background_subtracted_image_path is None
                ):
                    return False
                path = self.background_subtracted_image_path[
                    epoch_id, filter_type, stacking_method
                ]
            case PipelineProductType.qh2o_vs_aperture_radius:
                if (
                    epoch_id is None
                    or stacking_method is None
                    or self.qh2o_vs_aperture_radius_path is None
                ):
                    return False
                path = self.qh2o_vs_aperture_radius_path[epoch_id, stacking_method]
            case PipelineProductType.qh2o_from_profile:
                if (
                    epoch_id is None
                    or stacking_method is None
                    or self.qh2o_from_profile_path is None
                ):
                    return False
                path = self.qh2o_from_profile_path[epoch_id, stacking_method]

        return path.exists()

    def get_product_path(
        self,
        p: PipelineProductType,
        epoch_id: Optional[PipelineEpochID] = None,
        filter_type: Optional[SwiftFilter] = None,
        stacking_method: Optional[StackingMethod] = None,
    ) -> Optional[pathlib.Path]:
        match p:  # type: ignore
            case PipelineProductType.observation_log:
                path = self.observation_log_path
            case PipelineProductType.comet_orbital_data:
                path = self.comet_orbital_data_path
            case PipelineProductType.earth_orbital_data:
                path = self.earth_orbital_data_path
            case PipelineProductType.epoch:
                if self.epoch_paths is None or epoch_id is None:
                    return None
                path = self.epoch_paths[epoch_id]
            case PipelineProductType.stacked_epoch:
                if self.stacked_epoch_path is None or epoch_id is None:
                    return None
                path = self.stacked_epoch_path[epoch_id]
            case PipelineProductType.stacked_image:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.stacked_image_path is None
                ):
                    return None
                path = self.stacked_image_path[epoch_id, filter_type, stacking_method]
            case PipelineProductType.stacked_image_header:
                return None
            case PipelineProductType.background_analysis:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.background_analysis_path is None
                ):
                    return None
                path = self.background_analysis_path[
                    epoch_id, filter_type, stacking_method
                ]
            case PipelineProductType.background_subtracted_image:
                if (
                    epoch_id is None
                    or filter_type is None
                    or stacking_method is None
                    or self.background_subtracted_image_path is None
                ):
                    return None
                path = self.background_subtracted_image_path[
                    epoch_id, filter_type, stacking_method
                ]
            case PipelineProductType.qh2o_vs_aperture_radius:
                if (
                    epoch_id is None
                    or stacking_method is None
                    or self.qh2o_vs_aperture_radius_path is None
                ):
                    return None
                path = self.qh2o_vs_aperture_radius_path[epoch_id, stacking_method]
            case PipelineProductType.qh2o_from_profile:
                if (
                    epoch_id is None
                    or stacking_method is None
                    or self.qh2o_from_profile_path is None
                ):
                    return None
                path = self.qh2o_from_profile_path[epoch_id, stacking_method]

        return path

    def _delete_analysis_qh2o_from_profile_products(self) -> None:
        if self.epoch_ids is None or self.qh2o_from_profile_path is None:
            return
        for epoch_id, stacking_method in product(
            self.epoch_ids, [StackingMethod.summation, StackingMethod.median]
        ):
            p = self.qh2o_from_profile_path[epoch_id, stacking_method]
            if p.exists():
                p.unlink()

    def _delete_analysis_qh2o_vs_aperture_radius_products(self) -> None:
        if self.epoch_ids is None or self.qh2o_vs_aperture_radius_path is None:
            return
        for epoch_id, stacking_method in product(
            self.epoch_ids, [StackingMethod.summation, StackingMethod.median]
        ):
            p = self.qh2o_vs_aperture_radius_path[epoch_id, stacking_method]
            if p.exists():
                p.unlink()

    def _delete_analysis_bg_subtracted_image_products(self) -> None:
        if self.epoch_ids is None or self.background_subtracted_image_path is None:
            return
        for epoch_id, filter_type, stacking_method in product(
            self.epoch_ids,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            p = self.background_subtracted_image_path[
                epoch_id, filter_type, stacking_method
            ]
            if p.exists():
                p.unlink()

    def _delete_analysis_background_products(self) -> None:
        if self.epoch_ids is None or self.background_analysis_path is None:
            return
        for epoch_id, filter_type, stacking_method in product(
            self.epoch_ids,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            p = self.background_analysis_path[epoch_id, filter_type, stacking_method]
            if p.exists():
                p.unlink()

    def _delete_stacked_image_products(self) -> None:
        if self.epoch_ids is None or self.stacked_image_path is None:
            return
        for epoch_id, filter_type, stacking_method in product(
            self.epoch_ids,
            [SwiftFilter.uw1, SwiftFilter.uvv],
            [StackingMethod.summation, StackingMethod.median],
        ):
            p = self.stacked_image_path[epoch_id, filter_type, stacking_method]
            if p.exists():
                p.unlink()

    def _delete_stacked_epoch_products(self) -> None:
        if self.epoch_ids is None or self.stacked_epoch_path is None:
            return
        for epoch_id in self.epoch_ids:
            p = self.stacked_epoch_path[epoch_id]
            if p.exists():
                p.unlink()

    def _delete_epoch_products(self) -> None:
        if self.epoch_ids is None or self.epoch_paths is None:
            return
        for epoch_id in self.epoch_ids:
            p = self.epoch_paths[epoch_id]
            if p.exists():
                p.unlink()

    def delete_epochs_and_their_results(self):
        self._delete_analysis_qh2o_from_profile_products()
        self._delete_analysis_qh2o_vs_aperture_radius_products()
        self._delete_analysis_bg_subtracted_image_products()
        self._delete_analysis_background_products()
        self._delete_stacked_image_products()
        self._delete_stacked_epoch_products()
        self._delete_epoch_products()
