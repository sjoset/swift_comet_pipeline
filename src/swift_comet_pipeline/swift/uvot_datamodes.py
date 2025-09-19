import pathlib


from swift_comet_pipeline.scp_types.primitive import *


def datamode_from_fits_keyword_string(
    datamode: str, fits_file_path: pathlib.Path
) -> UvotImageMode | None:
    if datamode == "IMAGE":
        return UvotImageMode.data_mode
    elif datamode == "EVENT":
        return UvotImageMode.event_mode
    else:
        # Alternatively, we can just ask the user here
        print(
            f"Invalid data mode string: [{datamode}]! Inferring from path {fits_file_path}."
        )
        uvot_folder_path = fits_file_path.parent.parent
        event_folder_path = uvot_folder_path / pathlib.Path("event")
        print(f"Testing existence of {event_folder_path} ...")
        if event_folder_path.exists():
            return UvotImageMode.event_mode
        else:
            return UvotImageMode.data_mode


def datamode_to_pixel_resolution(datamode: UvotImageMode) -> UvotPixelResolution:
    if datamode == UvotImageMode.data_mode:
        return UvotPixelResolution.data_mode
    elif datamode == UvotImageMode.event_mode:
        return UvotPixelResolution.event_mode


def pixel_resolution_to_datamode(pixel_res: UvotPixelResolution) -> UvotImageMode:
    if pixel_res == UvotPixelResolution.data_mode:
        return UvotImageMode.data_mode
    elif pixel_res == UvotPixelResolution.event_mode:
        return UvotImageMode.event_mode


def float_to_pixel_resolution(pixel_float: float) -> UvotPixelResolution | None:
    if pixel_float == UvotPixelResolution.data_mode:
        return UvotPixelResolution.data_mode
    elif pixel_float == UvotPixelResolution.event_mode:
        return UvotPixelResolution.event_mode
    else:
        return None
