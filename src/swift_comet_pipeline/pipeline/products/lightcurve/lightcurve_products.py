import pathlib

from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.pipeline.products.product_io_types.csv_product import (
    CSVDataframePipelineProductIO,
)


# TODO: light curve products should depend on the stacking method!
class LightCurveProduct(PipelineProduct):

    def __init__(self, product_path: pathlib.Path):
        super().__init__(product_path=product_path)
        self.product_path = self.product_path / "lightcurves"
        self.product_path.mkdir(parents=True, exist_ok=True)


class CompleteVectorialLightCurveProduct(
    CSVDataframePipelineProductIO, LightCurveProduct
):
    # TODO: describe columns expected in the dataframe?
    """
    Contains production values resulting from different vectorial model fitting methods at varying dust redness
    """

    def __init__(self, product_path: pathlib.Path):
        super().__init__(product_path=product_path)

        self.product_path = self.product_path / pathlib.Path(
            "lightcurves_complete_vectorial.csv"
        )


class BestRednessLightCurveProduct(CSVDataframePipelineProductIO, LightCurveProduct):
    # TODO: describe columns expected in the dataframe?
    """
    Contains production values resulting from the specified fitting method, using values that result from a dust redness that
    gives the lowest percent error in the production
    """

    def __init__(self, product_path: pathlib.Path, fit_type: VectorialFitType):
        super().__init__(product_path=product_path)

        self.product_path = self.product_path / pathlib.Path(
            f"best_lightcurve_{fit_type}.csv"
        )
