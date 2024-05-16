from astropy.io import fits

from swift_comet_pipeline.pipeline.products.pipeline_product import PipelineProduct


class FitsImageProductIO(PipelineProduct):
    """
    instead of fighting fits file formatting and HDU lists, we assert that hdu[0] is an empty primary HDU
    and that hdu[1] is an ImageHDU, with relevant header and image data, for the images we create via
    the pipeline

    This is NOT a general-purpose FITS reader/writer!
    """

    def write(self) -> None:
        super().write()
        if self._data is not None:
            self._data.writeto(self.product_path, overwrite=True)

    def read(self) -> None:
        hdul = fits.open(self.product_path, lazy_load_hdus=False, memmap=True)
        self._data = fits.ImageHDU(data=hdul[1].data, header=hdul[1].header)  # type: ignore
        hdul.close()
