import scipy

from swift_comet_pipeline.scp_types.primitive import *


def shift_image_keep_dimensions(
    img: SwiftUvotImage, shift_x: int, shift_y: int, fill_value=0
) -> SwiftUvotImage:
    """
    Moves the pixels of the image by shift_x pixels and shift_y pixels, while keeping the resulting image the same size.
    Newly created pixels are filled with fill_value.
    """

    shifted_img = scipy.ndimage.shift(
        img,
        shift=(shift_y, shift_x),
        order=0,
        mode="constant",
        cval=fill_value,
        prefilter=False,
    )
    return shifted_img


def shift_img_and_fill(
    img: np.ndarray, shift: int, axis: int, fill_pixel: float
) -> np.ndarray:
    """
    Rolls the image along a dimension using numpy's roll(), which copies pixels from the scrolled-off edge
    to the other side.
    We fill those copied pixels instead with fill_pixel, so that we have a shifted image with no wrapping around
    Original image should remain unaltered.
    """

    if shift == 0:
        return img.copy()
    _i = np.roll(a=img, shift=shift, axis=axis)
    if axis == 1:
        # x axis filling
        if shift > 0:
            _i[:, :shift] = fill_pixel
        elif shift < 0:
            _i[:, shift:] = fill_pixel
    elif axis == 0:
        # y axis filling
        if shift > 0:
            _i[:shift, :] = fill_pixel
        elif shift < 0:
            _i[shift:, :] = fill_pixel
    else:
        print(f"Invalid value for axis: {axis}")
        print(f"Image shape: {np.shape(img)}")
        return img.copy()
    return _i


def scroll_image(
    img, x_shift: int, y_shift: int, fill_pixel: float = 0.0
) -> np.ndarray:
    """
    See shift_img_and_fill() for details - convenience function for scrolling image along x and y
    with no pixel wrapping.
    Original image should remain unaltered.
    """
    _i = shift_img_and_fill(img=img, shift=x_shift, axis=1, fill_pixel=fill_pixel)
    _i = shift_img_and_fill(img=_i, shift=y_shift, axis=0, fill_pixel=fill_pixel)
    return _i
