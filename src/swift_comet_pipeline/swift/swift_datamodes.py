import pathlib

from swift_comet_pipeline.types.swift_image_mode import SwiftImageMode
from swift_comet_pipeline.types.swift_pixel_resolution import SwiftPixelResolution


def datamode_from_fits_keyword_string(
    datamode: str, fits_file_path: pathlib.Path
) -> SwiftImageMode | None:
    if datamode == "IMAGE":
        return SwiftImageMode.data_mode
    elif datamode == "EVENT":
        return SwiftImageMode.event_mode
    else:
        # Alternatively, we can just ask the user here
        print(
            f"Invalid data mode string: [{datamode}]! Inferring from path {fits_file_path}."
        )
        uvot_folder_path = fits_file_path.parent.parent
        event_folder_path = uvot_folder_path / pathlib.Path("event")
        print(f"Testing existence of {event_folder_path} ...")
        if event_folder_path.exists():
            return SwiftImageMode.event_mode
        else:
            return SwiftImageMode.data_mode


def datamode_to_pixel_resolution(datamode: SwiftImageMode) -> SwiftPixelResolution:
    if datamode == SwiftImageMode.data_mode:
        return SwiftPixelResolution.data_mode
    elif datamode == SwiftImageMode.event_mode:
        return SwiftPixelResolution.event_mode


def pixel_resolution_to_datamode(pixel_res: SwiftPixelResolution) -> SwiftImageMode:
    if pixel_res == SwiftPixelResolution.data_mode:
        return SwiftImageMode.data_mode
    elif pixel_res == SwiftPixelResolution.event_mode:
        return SwiftImageMode.event_mode


def float_to_pixel_resolution(pixel_float: float) -> SwiftPixelResolution | None:
    if pixel_float == SwiftPixelResolution.data_mode:
        return SwiftPixelResolution.data_mode
    elif pixel_float == SwiftPixelResolution.event_mode:
        return SwiftPixelResolution.event_mode
    else:
        return None
