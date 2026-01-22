import numpy as np


def match_color_scales(
    img1, img2, sprite_value_1: float | None = None, sprite_value_2: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rescales the pixels in both images to share the same color scale.

    The variables sprite_value_1 and sprite_value_2 indicate which pixels values should
    be shifted to zero (we treat zero as the sprite background elsewhere for compositing images).

    After that shift, the pixel values are re-scaled to have the same maximum for sharing
    a common color scale.

    If no argument for the sprite values are given, they are taken from the corner of the images at img[0, 0].
    """

    if not sprite_value_1:
        sprite_value_1 = img1[0, 0]
    if not sprite_value_2:
        sprite_value_2 = img2[0, 0]

    # shift the 'dead' color to zero
    new1 = img1 - sprite_value_1
    new2 = img2 - sprite_value_2

    # match color scales so they can share the same image without destroying color
    new2 = np.max(new1) * (new2 / np.max(new2))

    return new1, new2


def match_color_scales_multi(img_list, reference_img) -> list[np.ndarray]:
    # reference_img_cp = reference_img.copy() - reference_img[0, 0]
    reference_max = np.max(reference_img) - reference_img[0, 0]

    # shift the 'dead' color to zero
    new_imgs = [i - i[0, 0] for i in img_list]

    # shift the max to be the max of the reference image
    altered_list = [(reference_max / np.max(i)) * i for i in new_imgs]
    # return_list.extend(altered_list)

    return altered_list
