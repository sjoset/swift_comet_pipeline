from astropy.coordinates import Angle
import astropy.units as u


def swift_position_angle(image_angle: Angle) -> Angle:
    # take an image_angle measured from the positive x-axis and turn into a position angle
    # level 2 images are rotated so that north is "up" and east is "left"
    # PA is measured from north counter-clockwise, so this is equivalent to subtracting 90 degrees
    # from the image angle

    pa = image_angle - (90 * u.deg)  # type: ignore

    wrapped_pa = Angle(angle=pa).wrap_at(360 * u.deg)  # type: ignore
    assert wrapped_pa is not None

    return wrapped_pa
