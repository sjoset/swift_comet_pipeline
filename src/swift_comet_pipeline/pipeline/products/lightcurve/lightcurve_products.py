import pathlib

from swift_comet_pipeline.modeling.vectorial_model_fit_type import VectorialFitType
from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct
from swift_comet_pipeline.pipeline.products.product_io_types.csv_product import (
    CSVDataframePipelineProductIO,
)
from swift_comet_pipeline.stacking.stacking_method import StackingMethod


class LightCurveProduct(PipelineProduct):

    def __init__(self, product_path: pathlib.Path, stacking_method: StackingMethod):
        super().__init__(product_path=product_path)
        self.product_path = self.product_path / "lightcurves"
        self.product_path.mkdir(parents=True, exist_ok=True)
        self.stacking_method = stacking_method


class CompleteVectorialLightCurveProduct(
    CSVDataframePipelineProductIO, LightCurveProduct
):
    # TODO: describe columns expected in the dataframe?
    """
    Contains production values resulting from different vectorial model fitting methods at varying dust redness
    """

    def __init__(self, product_path: pathlib.Path, stacking_method: StackingMethod):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"lightcurves_complete_vectorial_{self.stacking_method}.csv"
        )


class BestRednessLightCurveProduct(CSVDataframePipelineProductIO, LightCurveProduct):
    # TODO: describe columns expected in the dataframe?
    """
    Contains production values resulting from the specified fitting method, using values that result from a dust redness that
    gives the lowest percent error in the production
    """

    def __init__(
        self,
        product_path: pathlib.Path,
        stacking_method: StackingMethod,
        fit_type: VectorialFitType,
    ):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"best_lightcurve_{fit_type}_{self.stacking_method}.csv"
        )
