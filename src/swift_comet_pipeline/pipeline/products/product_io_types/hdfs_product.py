import pandas as pd

from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


class HDF5DataframePipelineProductIO(PipelineProduct):
    """
    Product for pd.DataFrame <----> hdf file

    Any metadata attached to the dataframe in the dict self._data.attrs will also be preserved
    """

    def read(self) -> None:
        super().read()
        with pd.HDFStore(self.product_path) as hdf:
            self._data = hdf["main_dataframe"]
            metadata = hdf.get_storer("main_dataframe").attrs.metadata
            self._data.attrs.update(metadata)

    def write(self) -> None:
        super().write()
        if self._data is None:
            return

        with pd.HDFStore(self.product_path) as hdf:
            hdf.put("main_dataframe", self._data, format="table", complevel=9)
            hdf.get_storer("main_dataframe").attrs.metadata = self._data.attrs
