import pathlib

from swift_comet_pipeline.observationlog.observation_log import (
    read_observation_log,
    write_observation_log,
)
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


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
