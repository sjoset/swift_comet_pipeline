import pathlib

from swift_comet_pipeline.pipeline.products.product_io_types.csv_product import (
    CSVDataframePipelineProductIO,
)


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
