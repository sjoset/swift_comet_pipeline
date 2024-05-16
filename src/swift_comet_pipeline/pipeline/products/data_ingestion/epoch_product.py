from swift_comet_pipeline.observationlog.epoch import read_epoch, write_epoch
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


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
