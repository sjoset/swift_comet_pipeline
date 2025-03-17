import numpy as np

from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution


class CoincidenceCorrection:

    kernel = {}

    def __init__(self) -> None:
        # Poole 2008
        # polynomial coefficients
        self.a1: float = -0.0663428
        self.a2: float = 0.0900434
        self.a3: float = -0.0237695
        self.a4: float = -0.0336789
        # dead time, seconds
        self.dead_time: float = 0.01577
        self.alpha = 1 - self.dead_time
        # frame time, seconds
        self.frame_time: float = 0.0110329

        self.poly = np.poly1d([self.a4, self.a3, self.a2, self.a1, 1])

        if not CoincidenceCorrection.kernel:
            for scale in [
                SwiftPixelResolution.data_mode,
                SwiftPixelResolution.event_mode,
            ]:
                self.kernel[scale] = self.make_kernel(image_scale=scale)

    def coi_factor(self, raw_pixel_count_rate: float) -> float:
        x = self.alpha * raw_pixel_count_rate * self.frame_time

        # calculate theoretical count rate
        cr_theory = -np.log(1 - x) / (self.alpha * self.frame_time)

        # corrected count rate
        cr_corrected = cr_theory / np.polyval(self.poly, x)

        ratio = cr_corrected / raw_pixel_count_rate
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=0.0, neginf=0.0)

        return ratio

    def make_kernel(self, image_scale: SwiftPixelResolution) -> np.ndarray:
        """
        Make a gaussian psf, returned as a 2d array, with the gaussian peak in the middle of the array in both dimensions
        """
        kernel_radius = np.ceil(5.0 / float(image_scale)).astype(np.int32)
        # the edge of our psf will encapsulate up to 3 sigma of the psf
        sigma = kernel_radius / 3.0
        x = np.arange(-kernel_radius // 2 + 1, kernel_radius // 2 + 1)
        y = np.arange(-kernel_radius // 2 + 1, kernel_radius // 2 + 1)
        x, y = np.meshgrid(x, y)
        psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        # normalize the psf
        psf /= psf.sum()
        return psf
