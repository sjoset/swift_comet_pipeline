import pathlib
from abc import ABC, abstractmethod
from typing import Any

from icecream import ic


class PipelineProduct(ABC):
    """
    Base class for files produced by the pipeline - given a path product_path, this object is responsible
    for loading and saving data
    """

    def __init__(self, product_path: pathlib.Path):
        self.product_path = product_path
        self._data: Any | None = None

    @property
    def data(self) -> Any | None:
        return self._data

    @data.setter
    def data(self, val: Any | None):
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

    # TODO:
    # seems goofy to do it this way - rename read() and write() to read_data and write_data, and this to read() ?
    # possibly an option to force reading in case we ever need to refresh data from disk
    def read_product_if_not_loaded(self):
        if self.product_path.exists() is False:
            return

        if self._data is None:
            self.read()
