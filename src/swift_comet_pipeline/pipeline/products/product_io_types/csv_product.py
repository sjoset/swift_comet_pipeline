import pandas as pd

from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


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
