import numpy as np


def build_composite_image(
    sprite_key: float, sprites: list[np.ndarray], x_offsets: list[int]
) -> np.ndarray:
    # takes list of sprites of equal height and paste them into a canvas at positive x_offsets with sprite_key for transparency

    # compute the width required for the composite image by pasting sprite 's' at offset 'x_off' - then take the largest width to build our canvas
    ws = [x_off + s.shape[1] for x_off, s in zip(x_offsets, sprites)]

    composite_w = np.max(ws)
    composite_h = sprites[0].shape[0]

    # transparent-colored canvas
    canvas = np.full((composite_h, composite_w), sprite_key)

    for s, x_off in zip(sprites, x_offsets):
        s_w = s.shape[1]
        pixel_mask = s != sprite_key
        paste_region = canvas[:, x_off : x_off + s_w]
        paste_region[pixel_mask] = s[pixel_mask]

    return canvas
