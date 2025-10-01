from swift_comet_pipeline.scp_types.primitive.pixel_coord import PixelCoord


# this is so ugly but it seems to work
def get_event_mode_pixel_offset() -> PixelCoord:
    """
    After time-slicing an event mode image, shift every pixel.  The image is assumed to still be an event mode 0.5 arcsec/pixel scale.
    """
    return PixelCoord(x=-12, y=0)
