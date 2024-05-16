import yaml

from icecream import ic

from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


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
