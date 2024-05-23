from astropy.table import Table
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


class ECSVDataframePipelineProductIO(PipelineProduct):
    """
    Product for pd.DataFrame <----> ecsv file to preserve metadata stored in the pandas dataframe .attrs dictionary
    """

    def read(self) -> None:
        super().read()
        t = Table.read(self.product_path, format="ascii.ecsv")
        self._data = t.to_pandas()
        self._data.attrs.update(t.meta)

    def write(self) -> None:
        super().write()
        if self._data is not None:
            t = Table.from_pandas(self._data)
            t.meta = self._data.attrs
            t.write(self.product_path, format="ascii.ecsv", overwrite=True)
