import numpy as np


def build_composite_image(
    sprite_key: float,
    sprites: list[np.ndarray],
    x_offsets: list[int],
    blend_images: list[bool] | None = None,
) -> np.ndarray:
    # takes list of sprites of equal height and paste them into a canvas at positive x_offsets with sprite_key for transparency
    # the z-ordering is from first image to last: an image is overwritten by later images in the list
    # if blend_images is True, then the pixels are pasted in as successive averages in their z-order

    if blend_images is None:
        blend_images = [False] * len(x_offsets)

    # compute the width required for the composite image by pasting sprite 's' at offset 'x_off' - then take the largest width to build our canvas
    ws = [x_off + s.shape[1] for x_off, s in zip(x_offsets, sprites)]

    composite_w = np.max(ws)
    composite_h = sprites[0].shape[0]

    # transparent-colored canvas
    canvas = np.full((composite_h, composite_w), sprite_key)

    for s, x_off, blend in zip(sprites, x_offsets, blend_images):
        s_w = s.shape[1]
        pixel_mask = s != sprite_key
        paste_region = canvas[:, x_off : x_off + s_w]
        if blend:
            # if we are pasting onto blank canvas, don't average - that distorts the images by (pixel + sprite_key)/2
            non_average_mask = np.logical_and(pixel_mask, paste_region == sprite_key)
            average_mask = np.logical_and(pixel_mask, paste_region != sprite_key)

            paste_region[non_average_mask] = s[non_average_mask]
            paste_region[average_mask] = s[average_mask] + paste_region[average_mask]
        else:
            # no blending, just bitblt
            paste_region[pixel_mask] = s[pixel_mask]

    return canvas
