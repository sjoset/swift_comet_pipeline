import pathlib
import glob
import calendar
from typing import List

import numpy as np
from astropy.time import Time

from swift_comet_pipeline.observationlog.epoch import Epoch
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


class DataIngestionFiles:
    """
    The data ingestion step consists of:
    - gathering all of the data from swift-produced FITS files into an observation log database
    - splitting the observation log into epochs interactively
    - preparing the epochs for analysis by allowing manual review/veto of all of the images
    - comet center adjustment in case the FITS files contain incorrect headers, or Horizons does not have a perfect orbital solution
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

        self.epochs: List[EpochProduct] | None = None
        self.scan_for_epoch_files()

    def _find_epoch_files(self) -> List[pathlib.Path] | None:
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
        self.epoch_dir_path.mkdir(exist_ok=True)
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
