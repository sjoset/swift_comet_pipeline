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
    """
    Contains production values resulting from different vectorial model fitting methods at varying dust redness
    This is a concatenation of near, far, and full fits

    columns:
    observation_time, time_from_perihelion_days, rh_au, near_fit_q, near_fit_q_err, far_fit_q, far_fit_q_err, full_fit_q, full_fit_q_err, dust_redness
    """

    def __init__(self, product_path: pathlib.Path, stacking_method: StackingMethod):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"complete_vectorial_lightcurve_{self.stacking_method}.csv"
        )


class BestRednessLightCurveProduct(CSVDataframePipelineProductIO, LightCurveProduct):
    """
    Contains production values resulting from the specified fitting method, using values that result from a dust redness that
    gives the lowest percent error in the production

    columns:
    observation_time, time_from_perihelion_days, rh_au, q, q_err, dust_redness
    """

    def __init__(
        self,
        product_path: pathlib.Path,
        stacking_method: StackingMethod,
        fit_type: VectorialFitType,
    ):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"best_vectorial_lightcurve_{fit_type}_{self.stacking_method}.csv"
        )


# TODO: We have to pick the near, far, or full fit (or all of them) to apply bayesian dust analysis - this isn't enough!

# class BayesianVectorialLightCurveProduct(
#     CSVDataframePipelineProductIO, LightCurveProduct
# ):
#     """
#     Holds the result of taking CompleteVectorialLightCurveProduct and appyling various dust redness priors
#     """
#
#     # TODO: describe columns
#
#     def __init__(
#         self,
#         product_path: pathlib.Path,
#         stacking_method: StackingMethod,
#     ):
#         super().__init__(product_path=product_path, stacking_method=stacking_method)
#
#         self.product_path = self.product_path / pathlib.Path(
#             f"bayesian_vectorial_lightcurve_{self.stacking_method}.csv"
#         )


class ApertureLightCurveProduct(CSVDataframePipelineProductIO, LightCurveProduct):
    """
    Contains production values as a function of aperture radius at varying dust redness at every epoch, with the production being averages over all of the plateaus found at that redness
    """

    # TODO: describe columns

    def __init__(self, product_path: pathlib.Path, stacking_method: StackingMethod):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"aperture_lightcurve_{self.stacking_method}.csv"
        )


class BayesianApertureLightCurveProduct(
    CSVDataframePipelineProductIO, LightCurveProduct
):
    """
    Contains production values at various dust redness priors at every epoch: it is a further-processed version of the ApertureLightCurveProduct
    """

    # TODO: describe columns

    def __init__(self, product_path: pathlib.Path, stacking_method: StackingMethod):
        super().__init__(product_path=product_path, stacking_method=stacking_method)

        self.product_path = self.product_path / pathlib.Path(
            f"bayesian_aperture_lightcurve_{self.stacking_method}.csv"
        )
